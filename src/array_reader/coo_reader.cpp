
#include "coo_reader.hpp"

namespace duckdb {

void CopyToVector(LogicalType type, DataChunk &output, idx_t vecIdx, char *vals,
                  uint64_t current_filled, uint64_t local_remains,
                  idx_t dataLen, idx_t offset) {
    // TODO: support more types
    if (type == LogicalType::INTEGER) {
        auto vec = FlatVector::GetData<int32_t>(output.data[vecIdx]);
        for (uint64_t idx = 0; idx < local_remains; idx++) {
            vec[current_filled + idx] = *(int32_t *)(vals + (idx * dataLen) +
                                    offset);
            // fprintf(stderr, "\tvec[%ld] = %d\n", idx, vec[idx]);
        }
    } else if (type == LogicalType::FLOAT) {
        auto vec = FlatVector::GetData<float>(output.data[vecIdx]);
        for (uint64_t idx = 0; idx < local_remains; idx++) {
            vec[current_filled + idx] =
                *(float *)(vals + (idx * dataLen) + offset);
        }
    } else if (type == LogicalType::DOUBLE) {
        auto vec = FlatVector::GetData<double>(output.data[vecIdx]);
        for (uint64_t idx = 0; idx < local_remains; idx++) {
            vec[current_filled + idx] =
                *(double *)(vals + (idx * dataLen) + offset);
            // fprintf(stderr, "\tvec[%ld] = %lf\n", idx, vec[idx]);
        }
    } else {
        throw NotImplementedException("Unsupported type");
    }
}

uint64_t CooReader::_PutData(optional_ptr<const FunctionData> bind_data,
                             ArrayReadGlobalState &gstate, char *pagevals,
                             vector<uint64_t *> &coords, uint64_t num_rows,
                             DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    auto total_remains = num_rows - gstate.cell_idx;
    auto local_remains = std::min(
        (uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    // fprintf(stderr, "size: %ld\n", num_rows);

    // offsets for multi-attributes
    vector<idx_t> offsets;
    idx_t offset = 0;       // final value of it will be a size of a row
    for (uint32_t i = 0; i < data.attrTypes.size(); i++) {
        offsets.push_back(offset);
        offset += GetTypeIdSize(data.attrTypes[i].InternalType());
    }

    // for each column
    for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
        auto dest = gstate.column_ids[gstate.projection_ids[i]];
        // dimensions
        if (dest < coords.size()) {
            // fprintf(stderr, "d%lu\n", dest);
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t buf_idx = gstate.cell_idx + idx;
                auto lcoord = coords[dest][buf_idx];
                auto gcoord =
                    (uint32_t)lcoord +
                    (gstate.currentCoordsInTile[dest] * data.tile_size[dest]);
                vec[current_filled + idx] = gcoord;
                // fprintf(stderr, "\tvec[%ld] = %d\n", idx,
                //         vec[current_filled + idx]);
            }
        } else {
            // attributes
            char *vals = (char *)pagevals;
            int attrIdx = dest - coords.size();
            auto type = data.attrTypes[attrIdx];
            // fprintf(stderr, "a%d\n", attrIdx);
            CopyToVector(type, output, i, vals, current_filled, local_remains,
                         offset, offsets[attrIdx]);
        }
    }

    uint64_t produced = local_remains;
    gstate.cell_idx += produced;
    return produced;
}

uint64_t CooReader::_PutDataNoPrune(optional_ptr<const FunctionData> bind_data,
                                    ArrayReadGlobalState &gstate,
                                    char *pagevals,
                                    vector<uint64_t *> &coords,
                                    uint64_t num_rows, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    auto total_remains = num_rows - gstate.cell_idx;
    auto local_remains = std::min(
        (uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    // offsets for multi-attributes
    vector<idx_t> offsets;
    idx_t offset = 0;       // final value of it will be a size of a row
    for (uint32_t i = 0; i < data.attrTypes.size(); i++) {
        offsets.push_back(offset);
        offset += GetTypeIdSize(data.attrTypes[i].InternalType());
    }

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t buf_idx = gstate.cell_idx + idx;

        // uint32_t x = (uint32_t)pagevals[buf_idx];
        // uint32_t y = (uint32_t)pagevals[buf_idx + 1];

        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            auto colIdx = gstate.column_ids[i];

            // dimensions
            if (colIdx < coords.size()) {
                auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
                vec[current_filled + idx] = (uint32_t)coords[colIdx][buf_idx];
            } else {
                // attributes
                char *vals = (char *)pagevals;
                int attrIdx = colIdx - coords.size();
                auto type = data.attrTypes[attrIdx];
                CopyToVector(type, output, i, vals, current_filled,
                             local_remains, offset, offsets[attrIdx]);
            }
        }

        // std::cout << "\t[Put2DDataNoPrune] idx: " << idx << ", x: " <<
        // coords[0] << ", y: " << coords[1] << ", val: " << val <<
        // std::endl; free(coords);
    }

    uint64_t produced = local_remains;
    gstate.cell_idx += produced;
    return produced;
}

uint64_t CooReader::_PutDataNoPruneAndProjection(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    char *pagevals, vector<uint64_t *> &coords, uint64_t num_rows,
    DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    auto total_remains = num_rows - gstate.cell_idx;
    auto local_remains = std::min(
        (uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    // offsets for multi-attributes
    vector<idx_t> offsets;
    idx_t offset = 0;       // final value of it will be a size of a row
    for (uint32_t i = 0; i < data.attrTypes.size(); i++) {
        offsets.push_back(offset);
        offset += GetTypeIdSize(data.attrTypes[i].InternalType());
    }

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t buf_idx = gstate.cell_idx + idx;

        // iterate over columns
        for (uint32_t i = 0; i < output.ColumnCount(); i++) {
            // dimensions
            if (i < coords.size()) {
                auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
                vec[current_filled + idx] = (uint32_t)coords[i][buf_idx];
            } else {
                // attributes
                char *vals = (char *)pagevals;
                int attrIdx = i - coords.size();
                auto type = data.attrTypes[attrIdx];
                CopyToVector(type, output, i, vals, current_filled,
                             local_remains, offset, offsets[attrIdx]);
            }
        }

        // std::cout << "\t[Put2DDataNoPruneAndProjection] idx: " << idx <<
        // ", x: " << xs[idx] << ", y: " << ys[idx] << ", val: " <<
        // vals[idx] << std::endl; free(coords);
    }

    uint64_t produced = local_remains;
    gstate.cell_idx += produced;
    return produced;
}

uint64_t CooReader::PutData(optional_ptr<const FunctionData> bind_data,
                            ArrayReadGlobalState &gstate, char *pagevals,
                            vector<uint64_t *> &coords, uint64_t num_of_cells,
                            DataChunk &output) {
    // auto &data = bind_data->Cast<ArrayReadData>();
    if (gstate.projection_ids.size() > 0)  // filter_prune ON
        return _PutData(bind_data, gstate, pagevals, coords, num_of_cells,
                        output);
    else if (gstate.column_ids.size() == output.data.size())  // no filter prune
        return _PutDataNoPrune(bind_data, gstate, pagevals, coords,
                               num_of_cells, output);
    else  // projection_pushdown and filter_prune are both false
        return _PutDataNoPruneAndProjection(bind_data, gstate, pagevals, coords,
                                            num_of_cells, output);
}
}  // namespace duckdb