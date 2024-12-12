#pragma once

#include <chrono>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <thread>

#include "array_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"

#include "buffer/bf.h"
#include "io/tilestore.h"

namespace duckdb {

struct ArrayReadData : public TableFunctionData {
   public:
    ArrayReadData(string arrayname) {
        uint64_t **_dim_domains;
        uint64_t *_tile_size;
        uint64_t *_array_size_in_tile;
        tilestore_datatype_t *_attr_types;
        uint32_t _num_attrs;
        storage_util_get_array_type(arrayname.c_str(), &format);
        storage_util_get_dim_domains(arrayname.c_str(), &_dim_domains,
                                     &dim_len);
        storage_util_get_tile_extents(arrayname.c_str(), &_tile_size, &dim_len);
        storage_util_get_dcoord_lens(_dim_domains, _tile_size, dim_len,
                                     &_array_size_in_tile);
        storage_util_get_attribute_type(arrayname.c_str(), "", &_attr_types,
                                        &_num_attrs);

        array_size_in_tile = vector<uint64_t>(_array_size_in_tile,
                                              _array_size_in_tile + dim_len);
        tile_size = vector<uint64_t>(_tile_size, _tile_size + dim_len);

        this->arrayname = arrayname;

        num_cells = 1;
        for (uint32_t i = 0; i < dim_len; i++) {
            num_cells *= tile_size[i];
        }

        for (uint32_t i = 0; i < _num_attrs; i++) {
            if (_attr_types[i] == TILESTORE_INT32) {
                attrTypes.push_back(LogicalType::INTEGER);
            } else if (_attr_types[i] == TILESTORE_FLOAT32) {
                attrTypes.push_back(LogicalType::FLOAT);
            } else if (_attr_types[i] == TILESTORE_FLOAT64) {
                attrTypes.push_back(LogicalType::DOUBLE);
            } else {
                throw NotImplementedException("Unsupported attribute type: " +
                                              std::to_string(_attr_types[i]));
            }
        }

        storage_util_free_dim_domains(&_dim_domains, dim_len);
        storage_util_free_tile_extents(&_tile_size, dim_len);
        storage_util_free_dcoord_lens(&_array_size_in_tile);
        free(_attr_types);
    };

   public:
    string arrayname;
    vector<uint64_t> array_size_in_tile;
    vector<uint64_t> tile_size;
    uint64_t num_cells;
    uint32_t dim_len;
    tilestore_format_t format;

    // requested coords
    // it will be set when binding. if empty, read all
    vector<int> requestedCoords;

    // attribute types if multi-attributes
    vector<LogicalType> attrTypes;
};

struct ArrayReadGlobalState : public GlobalTableFunctionState {
   public:
    ArrayReadGlobalState(ClientContext &context,
                         TableFunctionInitInput &input) {
        auto &data = input.bind_data->Cast<ArrayReadData>();

        isFinished = false;
        dimLen = data.dim_len;
        page = NULL;
        for (int d = 0; d < (int)dimLen; d++) {
            arrSizeInTile.push_back((int)data.array_size_in_tile[d]);
        }

        auto arrname = data.arrayname.c_str();
        _arrname_char = new char[1024];  // for char*
        strcpy(_arrname_char, arrname);

        // copy
        column_ids.assign(input.column_ids.begin(), input.column_ids.end());
        projection_ids.assign(input.projection_ids.begin(),
                              input.projection_ids.end());

        if (data.requestedCoords.size() > 0) {
            currentCoordsInTile = data.requestedCoords;
            pin();
            if (page == NULL) {
                isFinished = true;
                unpin();
            }
        } else {
            currentCoordsInTile = vector<int>(data.dim_len, 0);
            --currentCoordsInTile[data.dim_len - 1];
            findNextTile();
        }

        ArrayExtension::ResetPVBufferStats();
    };

    bool findNextTile() {
        // find the next tile; this involves pin / unpin

        // loop until find a non-empty tile
        bool searchDone = false;
        while (!searchDone) {
            unpin();

            // find next coordinates and check if finished
            for (int idx = (int)dimLen - 1; idx >= 0; idx--) {
                currentCoordsInTile[idx] += 1;
                if (currentCoordsInTile[idx] < arrSizeInTile[idx]) {
                    break;
                }

                currentCoordsInTile[idx] = 0;
                if (idx == 0) {
                    // if hit here, all tiles have been searched
                    isFinished = true;
                    searchDone = true;
                }
            }

            // if found a tile 
            if (!isFinished) {
                // reset cell index and pin the tile
                cell_idx = 0;
                pin();
                if (page == NULL) {
                    // if the tile is empty, keep searching
                    continue;
                } else {
                    // found!
                    searchDone = true;
                }
            } 
        }

        // std::cout << "coords=" << currentCoordsInTile[0] << "," << currentCoordsInTile[1] << "," << currentCoordsInTile[2] << std::endl;
        return !isFinished;
    }


    // FIXME: Why ArrayReadGlobalState() called multiple times?
    ~ArrayReadGlobalState() {}

    void free() {
        unpin();
        // delete _arrname_char;
        ArrayExtension::PrintPVBufferStats();
    };

    void pin() {
        if (page != NULL) {
            throw InternalException("Already pinned");
        }

        // get buffer
        auto dcoords = make_uniq_array<uint64_t>(dimLen);
        for (uint32_t idx = 0; idx < dimLen; idx++) {
            dcoords[idx] = (uint64_t) currentCoordsInTile[idx];
        }
        // TODO: Consider sparse tile in the future
        array_key key = {_arrname_char, dcoords.get(), dimLen,
                         BF_EMPTYTILE_NONE};

        if (BF_GetBuf(key, &page) != BFE_OK) {
            throw InternalException("Failed to get buffer");
        }
    }

    void unpin() {
        if (page == NULL) return;

        auto dcoords = make_uniq_array<uint64_t>(2);
        for (uint32_t idx = 0; idx < 2; idx++) {
            dcoords[idx] = (uint64_t) currentCoordsInTile[idx];
        }

        array_key key = {_arrname_char, dcoords.get(), 2, BF_EMPTYTILE_DENSE};
        BF_UnpinBuf(key);
        page = NULL;
    }

   public:
    uint64_t cell_idx;
    bool isFinished;
    PFpage *page;
    uint32_t dimLen;
    vector<int> currentCoordsInTile;
    vector<int> arrSizeInTile;

    // table function related:
    vector<column_t> column_ids;
    vector<idx_t> projection_ids;

   private:
    char *_arrname_char;
};

struct ArrayReadLocalState : public LocalTableFunctionState {
   public:
    ArrayReadLocalState(ClientContext &context, TableFunctionInitInput &input,
                        ArrayReadGlobalState &gstate) {};
};

}  // namespace duckdb