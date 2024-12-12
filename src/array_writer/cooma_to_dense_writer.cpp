#include "array_extension.hpp"
#include "array_writer.hpp"

#include "buffer/bf.h"

namespace duckdb {

GlobalCoomaToDenseWriteArrayData::GlobalCoomaToDenseWriteArrayData(
    ClientContext &context, FunctionData &bind_data, const string &file_path)
    : GlobalWriteArrayData(context, bind_data, file_path) {
    writer = make_uniq<CoomaToDenseCopyArrayWriter>(*this);
}

CoomaToDenseCopyArrayWriter::CoomaToDenseCopyArrayWriter(
    GlobalFunctionData &gstate) {
    auto &array_gstate = gstate.Cast<GlobalCoomaToDenseWriteArrayData>();
    array_gstate.isFirst = true;
}

void CoomaToDenseCopyArrayWriter::WriteArrayData(ExecutionContext &context,
                                               FunctionData &bind_data,
                                               GlobalFunctionData &gstate,
                                               LocalFunctionData &lstate,
                                               DataChunk &input) {
    // NOTE: I assume that only one thread runs

    // Get global state and data
    auto &array_gstate = gstate.Cast<GlobalCoomaToDenseWriteArrayData>();
    auto &array_data = bind_data.Cast<CopyArrayData>();

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
        uint64_t onedcoord = 0;
        int64_t base = 1;
        bool out = false;
        for (int64_t d = array_data.dim_len - 1; d >= 0; d--) {
            uint32_t *coord_vec = FlatVector::GetData<uint32_t>(input.data[d]);
            uint64_t lcoord = (uint64_t)coord_vec[i] % array_data.tile_size[d];
            tcoords[d] = (uint64_t)coord_vec[i] / array_data.tile_size[d];

            // std::cerr << "d[" << d << "]=" << lcoord << "(" << coord_vec[i] << "),";

            onedcoord += lcoord * base;
            base *= array_data.tile_size[d];

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
        }

        // write to the buffer
        char *target =
            (char *)array_gstate.buf + (onedcoord * array_gstate.dataLen);
        idx_t offset = 0;
        for (idx_t a = array_data.dim_len; a < array_gstate.dim_len; a++) {
            auto val = input.GetValue(a, i);
            auto size = GetTypeIdSize(val.type().InternalType());
            memcpy(target + offset, (void*) val.GetPointer(), size);
            offset += size;
        }

        // reset nullbit if nullable array
        if (array_gstate.page->type == DENSE_FIXED_NULLABLE ||
            array_gstate.page->type == SPARSE_FIXED_NULLABLE) {
            bf_util_reset_cell_null(array_gstate.page, onedcoord);
        }

        // std::cout << ", buf[" << idx << "] = " << val[i] << std::endl;
    }

    delete tcoords;
}

}  // namespace duckdb