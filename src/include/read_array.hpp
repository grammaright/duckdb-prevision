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

namespace duckdb {

struct ArrayReadData : public TableFunctionData {
   public:
    ArrayReadData(string arrayname) {
        uint64_t **_dim_domains;
        uint64_t *_tile_size;
        uint64_t *_array_size_in_tile;
        storage_util_get_dim_domains(arrayname.c_str(), &_dim_domains,
                                     &dim_len);
        storage_util_get_tile_extents(arrayname.c_str(), &_tile_size, &dim_len);
        storage_util_get_dcoord_lens(_dim_domains, _tile_size, dim_len,
                                     &_array_size_in_tile);

        array_size_in_tile = vector<uint64_t>(_array_size_in_tile,
                                              _array_size_in_tile + dim_len);
        tile_size = vector<uint64_t>(_tile_size, _tile_size + dim_len);

        // storage_util_free_dim_domains(&_dim_domains, dim_len);
        // storage_util_free_tile_extents(&_tile_size, dim_len);
        // storage_util_free_dcoord_lens(&_array_size_in_tile);

        this->arrayname = arrayname;

        num_cells = 1;
        for (uint32_t i = 0; i < dim_len; i++) {
            num_cells *= tile_size[i];
        }
    };

   public:
    string arrayname;
    vector<uint64_t> array_size_in_tile;
    vector<uint64_t> tile_size;
    uint64_t num_cells;
    uint32_t dim_len;

    bool is_coo_array = false;
};

struct ArrayReadGlobalState : public GlobalTableFunctionState {
   public:
    ArrayReadGlobalState(ClientContext &context,
                         TableFunctionInitInput &input) {
        auto &data = input.bind_data->Cast<ArrayReadData>();

        current_coords_in_tile = vector<uint64_t>(data.dim_len, 0);
        finished = false;

        // copy
        column_ids.assign(input.column_ids.begin(), input.column_ids.end());
        projection_ids.assign(input.projection_ids.begin(),
                              input.projection_ids.end());

        // get buffer
        auto dcoords = make_uniq_array<uint64_t>(2);
        for (uint32_t idx = 0; idx < data.dim_len; idx++) {
            dcoords[idx] = current_coords_in_tile[idx];
        }
        auto arrname = data.arrayname.c_str();
        _arrname_char = new char[1024];  // for char*
        strcpy(_arrname_char, arrname);

        // TODO: Consider sparse tile in the future
        array_key key = {_arrname_char, dcoords.get(), 2, BF_EMPTYTILE_DENSE};

        if (BF_GetBuf(key, &page) != BFE_OK) {
            throw InternalException("Failed to get buffer");
        }

        ArrayExtension::ResetPVBufferStats();
    };

    // FIXME: Why ArrayReadGlobalState() called multiple times?
    ~ArrayReadGlobalState() {}
    void free() {
        auto dcoords = make_uniq_array<uint64_t>(2);
        for (uint32_t idx = 0; idx < 2; idx++) {
            dcoords[idx] = current_coords_in_tile[idx];
        }

        array_key key = {_arrname_char, dcoords.get(), 2, BF_EMPTYTILE_DENSE};
        BF_UnpinBuf(key);
        // delete _arrname_char;

        ArrayExtension::PrintPVBufferStats();
    };

   public:
    uint64_t cell_idx;
    vector<uint64_t> current_coords_in_tile;
    bool finished;
    PFpage *page;

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