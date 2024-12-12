#include "array_extension.hpp"
#include "array_writer.hpp"

#include "buffer/bf.h"

namespace duckdb {

GlobalCOOToArrayWriteArrayData::GlobalCOOToArrayWriteArrayData(
    ClientContext &context, FunctionData &bind_data, const string &file_path)
    : GlobalWriteArrayData(context, bind_data, file_path) {
    writer = make_uniq<COOToArrayCopyArrayWriter>();
}

COOToArrayCopyArrayWriter::COOToArrayCopyArrayWriter() {}

void COOToArrayCopyArrayWriter::WriteArrayData(ExecutionContext &context,
                                               FunctionData &bind_data,
                                               GlobalFunctionData &gstate,
                                               LocalFunctionData &lstate,
                                               DataChunk &input) {
    // NOTE: I assume that only one thread runs
    auto &array_gstate = gstate.Cast<GlobalWriteArrayData>();
    auto &array_data = bind_data.Cast<CopyArrayData>();

    // We don't know what vector type DuckDB will give
    // So we need to convert it to unified vector format
    // vector type ref: https://youtu.be/bZOvAKGkzpQ?si=ShnWtUDKNIm7ymo8&t=1265

    // FIXME: Maybe performance panalty.
    // exploit the vector type
    input.Flatten();

    double *val = FlatVector::GetData<double>(input.data[array_data.dim_len]);

    uint64_t *tcoords = new uint64_t[array_data.dim_len];
    for (idx_t i = 0; i < input.size(); i++) {
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

        if (out) {
            // std::cout << ", unpinning";
            array_gstate.unpin();
        }

        if (!array_gstate.is_pinned) {
            // std::cout << ", pinning";
            array_gstate.pin(
                vector<uint64_t>(tcoords, tcoords + array_data.dim_len));
        }

        array_gstate.buf[onedcoord] = val[i];
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