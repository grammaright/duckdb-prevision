
#include "coo2tbl_converter.hpp"

namespace duckdb {
const uint64_t TOTAL_COO_NUM_COLUMNS = 3;

uint64_t COOToTableConverter::_PutData(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t num_rows, DataChunk &output) {
    // auto &data = bind_data->Cast<ArrayReadData>();

    auto total_remains =
        num_rows - (uint64_t)(gstate.cell_idx / TOTAL_COO_NUM_COLUMNS);
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE, total_remains);

    // for dest == 1 and 2
    for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
        auto dest = gstate.column_ids[gstate.projection_ids[i]];
        if (dest == 0 || dest == 1) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t buf_idx =
                    gstate.cell_idx + (idx * TOTAL_COO_NUM_COLUMNS + dest);
                vec[idx] = (uint32_t)pagevals[buf_idx];
            }
        } else if (dest == 2) {
            auto vec = FlatVector::GetData<double>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t buf_idx =
                    gstate.cell_idx + (idx * TOTAL_COO_NUM_COLUMNS + dest);
                vec[idx] = pagevals[buf_idx];
            }
        }
    }

    uint64_t produced = local_remains * TOTAL_COO_NUM_COLUMNS;
    gstate.cell_idx += produced;
    return local_remains;
}

uint64_t COOToTableConverter::_PutDataNoPrune(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t num_rows, DataChunk &output) {
    // auto &data = bind_data->Cast<ArrayReadData>();

    auto total_remains = num_rows - (gstate.cell_idx / TOTAL_COO_NUM_COLUMNS);
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE, total_remains);

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t buf_idx = gstate.cell_idx + (idx * TOTAL_COO_NUM_COLUMNS);

        uint32_t x = (uint32_t)pagevals[buf_idx];
        uint32_t y = (uint32_t)pagevals[buf_idx + 1];
        double val = pagevals[buf_idx + 2];

        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            if (gstate.column_ids[i] == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] = x;
            else if (gstate.column_ids[i] == 1)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] = y;
            else if (gstate.column_ids[i] == 2)
                FlatVector::GetData<double>(output.data[i])[idx] = val;
        }

        // std::cout << "\t[Put2DDataNoPrune] idx: " << idx << ", x: " <<
        // coords[0] << ", y: " << coords[1] << ", val: " << val <<
        // std::endl; free(coords);
    }

    uint64_t produced = local_remains * TOTAL_COO_NUM_COLUMNS;
    gstate.cell_idx += produced;
    return local_remains;
}

uint64_t COOToTableConverter::_PutDataNoPruneAndProjection(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t num_rows, DataChunk &output) {
    // auto &data = bind_data->Cast<ArrayReadData>();

    auto total_remains = num_rows - (gstate.cell_idx / TOTAL_COO_NUM_COLUMNS);
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE, total_remains);

    auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
    auto ys = FlatVector::GetData<uint32_t>(output.data[1]);
    auto vals =
        FlatVector::GetData<double>(output.data[2]);  // double type assumed

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t buf_idx = gstate.cell_idx + (idx * TOTAL_COO_NUM_COLUMNS);

        uint32_t x = (uint32_t)pagevals[buf_idx];
        uint32_t y = (uint32_t)pagevals[buf_idx + 1];
        double val = pagevals[buf_idx + 2];

        xs[idx] = x;
        ys[idx] = y;
        vals[idx] = val;

        // std::cout << "\t[Put2DDataNoPruneAndProjection] idx: " << idx <<
        // ", x: " << xs[idx] << ", y: " << ys[idx] << ", val: " <<
        // vals[idx] << std::endl; free(coords);
    }

    uint64_t produced = local_remains * TOTAL_COO_NUM_COLUMNS;
    gstate.cell_idx += produced;
    return local_remains;
}

uint64_t COOToTableConverter::PutData(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t num_of_cells, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    // COO must be 2D array
    assert(data.dim_len == 2);

    if (gstate.projection_ids.size() > 0)  // filter_prune ON
        return _PutData(bind_data, gstate, pagevals, num_of_cells, output);
    else if (gstate.column_ids.size() == output.data.size())  // no filter prune
        return _PutDataNoPrune(bind_data, gstate, pagevals, num_of_cells,
                               output);
    else  // projection_pushdown and filter_prune are both false
        return _PutDataNoPruneAndProjection(bind_data, gstate, pagevals,
                                            num_of_cells, output);
}
}  // namespace duckdb