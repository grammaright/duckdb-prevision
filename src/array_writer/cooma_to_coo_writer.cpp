#include "array_extension.hpp"
#include "array_writer.hpp"

#include "buffer/bf.h"
#include "buffer/utils.h"

namespace duckdb {

void writeToBuf(char *buf, Value val);

GlobalCoomaToCooWriteArrayData::GlobalCoomaToCooWriteArrayData(
    ClientContext &context, FunctionData &bind_data, const string &file_path)
    : GlobalWriteArrayData(context, bind_data, file_path) {
    writer = make_uniq<CoomaToCooCopyArrayWriter>(*this);
}

CoomaToCooCopyArrayWriter::CoomaToCooCopyArrayWriter(
    GlobalFunctionData &gstate) {
    auto &array_gstate = gstate.Cast<GlobalCoomaToCooWriteArrayData>();
    array_gstate.isFirst = true;
}

void CoomaToCooCopyArrayWriter::WriteArrayData(ExecutionContext &context,
                                               FunctionData &bind_data,
                                               GlobalFunctionData &gstate,
                                               LocalFunctionData &lstate,
                                               DataChunk &input) {
    // NOTE: I assume that only one thread runs

    // Get global state and data
    auto &array_gstate = gstate.Cast<GlobalCoomaToCooWriteArrayData>();
    auto &array_data = bind_data.Cast<CopyArrayData>();

    // Get tile data and coords
    auto page = array_gstate.page;
    void *vals;
    vector<uint64_t *> coords;
    if (page != NULL) {
        vals = (void *)bf_util_get_pagebuf(page);
        for (uint32_t i = 0; i < array_data.dim_len; i++) {
            auto coord = bf_util_pagebuf_get_coords(page, i);
            coords.push_back(coord);
        }
    } else {
        vals = NULL;
        for (uint32_t i = 0; i < array_data.dim_len; i++) {
            coords.push_back(NULL);
        }
    }

    // to set dataLen from DataChunk
    if (array_gstate.isFirst) {
        array_gstate.isFirst = false;

        array_gstate.dataLen = 0;
        for (idx_t a = 0; a < array_gstate.dim_len; a++) {
            auto size = GetTypeIdSize(input.GetTypes()[a].InternalType());
            array_gstate.dataLen += (int) size;
        }
    }

    // iterate over input table
    uint64_t *tcoords = new uint64_t[array_data.dim_len];   // reusing variable
    for (idx_t i = 0; i < input.size(); i++) {
        // calculate 1D coord in a tile and check if the cell is out of the tile
        bool out = false;
        for (int64_t d = array_data.dim_len - 1; d >= 0; d--) {
            uint32_t *coord_vec = FlatVector::GetData<uint32_t>(input.data[d]);
            // uint64_t lcoord = (uint64_t)coord_vec[i] % array_data.tile_size[d];
            tcoords[d] = (uint64_t)coord_vec[i] / array_data.tile_size[d];

            if (tcoords[d] != array_gstate.tile_coords[d]) {
                out = true;
            }
        }
        // std::cerr << " onedcoord=" << onedcoord << ", val=" << val[i] << std::endl;

        // unpin the tile if out of the tile
        if (out) {
            // std::cout << ", unpinning";
            array_gstate.unpin();
        }

        // get destination tile
        if (!array_gstate.is_pinned) {
            // std::cout << ", pinning";
            array_gstate.pin(
                vector<uint64_t>(tcoords, tcoords + array_data.dim_len));

            page = array_gstate.page;
            vals = (void *)bf_util_get_pagebuf(page);
            coords.clear();
            for (uint32_t i = 0; i < array_data.dim_len; i++) {
                auto coord = bf_util_pagebuf_get_coords(page, i);
                coords.push_back(coord);
            }
        }

        // write to the buffer
        // if the buffer is full, resize the buffer
        if (page->unfilled_idx >= page->max_idx) {
            BF_ResizeBuf(page, page->max_idx * 2);

            page = array_gstate.page;
            vals = (void *)bf_util_get_pagebuf(page);
            coords.clear();
            for (uint32_t i = 0; i < array_data.dim_len; i++) {
                auto coord = bf_util_pagebuf_get_coords(page, i);
                coords.push_back(coord);
            }
        }

        // data values
        char *target = (char *)vals + page->unfilled_pagebuf_offset;
        idx_t offset = 0;
        for (idx_t a = 0; a < input.ColumnCount(); a++) {
            auto val = input.GetValue(a, i);
            auto size = GetTypeIdSize(val.type().InternalType());
            if (a < array_gstate.dim_len) {
                // dimensions
                auto gcoord = val.GetValue<int64_t>();
                auto lcoord = gcoord % array_data.tile_size[a];
                ((uint64_t *)coords[a])[page->unfilled_idx] = lcoord;
            } else {
                // attributes
                writeToBuf(target + offset, val);
                offset += size;
            }
        }
        page->unfilled_pagebuf_offset += offset;
        page->unfilled_idx++;
        // std::cout << ", buf[" << idx << "] = " << val[i] << std::endl;
    }

    delete tcoords;
}

template <typename T>
void _writeToBuf(char *buf, T val) {
    memcpy(buf, &val, sizeof(T));
}

void writeToBuf(char *buf, Value val) {
    if (val.type() == LogicalType::INTEGER) {
        _writeToBuf<int32_t>(buf, val.GetValue<int32_t>());
    } else if (val.type() == LogicalType::BIGINT) {
        _writeToBuf<int64_t>(buf, val.GetValue<int64_t>());
    } else if (val.type() == LogicalType::FLOAT) {
        _writeToBuf<float>(buf, val.GetValue<float>());
    } else if (val.type() == LogicalType::DOUBLE) {
        _writeToBuf<double>(buf, val.GetValue<double>());
    } else {
        throw NotImplementedException("Unsupported type");
    }
}

}  // namespace duckdb