#include "array_extension.hpp"
#include "array_writer.hpp"

#include "buffer/bf.h"

namespace duckdb {

GlobalDenseToTileWriteArrayData::GlobalDenseToTileWriteArrayData(
    ClientContext &context, FunctionData &bind_data, const string &file_path)
    : GlobalWriteArrayData(context, bind_data, file_path) {
    auto &array_bind = bind_data.Cast<DenseToTileCopyArrayData>();
    writer = make_uniq<DenseToTileCopyArrayWriter>(*this, array_bind);
}

DenseToTileCopyArrayData::DenseToTileCopyArrayData(
    string file_path, ArrayCopyFunctionExecutionMode mode,
    vector<uint64_t> _tile_coords)
    : CopyArrayData(file_path, mode), tile_coords(_tile_coords) {}

DenseToTileCopyArrayWriter::DenseToTileCopyArrayWriter(
    GlobalFunctionData &gstate, DenseToTileCopyArrayData &array_data) {
    auto &array_gstate = gstate.Cast<GlobalWriteArrayData>();

    array_gstate.pin(array_data.tile_coords);

    cur_idx = 0;
}

void DenseToTileCopyArrayWriter::WriteArrayData(ExecutionContext &context,
                                                FunctionData &bind_data,
                                                GlobalFunctionData &gstate,
                                                LocalFunctionData &lstate,
                                                DataChunk &input) {
    // NOTE: I assume that only one thread runs
    auto &array_gstate = gstate.Cast<GlobalWriteArrayData>();

    // We don't know what vector type DuckDB will give
    // So we need to convert it to unified vector format
    // vector type ref: https://youtu.be/bZOvAKGkzpQ?si=ShnWtUDKNIm7ymo8&t=1265
    input.data[0].Flatten(input.size());  // FIXME: Maybe performance panalty.
                                          // exploit the vector type
    auto vector = FlatVector::GetData<double>(input.data[0]);

    D_ASSERT(cur_idx + input.size() <= array_gstate.buf_size);

    for (idx_t i = 0; i < input.size(); i++) {
        array_gstate.buf[cur_idx++] = vector[i];
    }
}

}  // namespace duckdb