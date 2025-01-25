
#include "csr_reader.hpp"

namespace duckdb {

uint64_t CsrReader::_PutData(optional_ptr<const FunctionData> bind_data,
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
        if (dest == 0) {
            // row column
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);

            // first, find the idxptrIdx of the current cell
            uint32_t idxptrIdx = 0;
            for (uint64_t idx = 0; idx < data.tile_size[1]; idx++) {
                if (coords[0][idx + 1] >= gstate.cell_idx) {
                    idxptrIdx = idx;
                    break;
                }
            }

            // fill the output vector
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t buf_idx = gstate.cell_idx + idx;
                // adjust idxptrIdx
                while (coords[0][idxptrIdx + 1] <= buf_idx) {
                    idxptrIdx++;
                }
                // put the value
                auto lcoord = idxptrIdx;
                auto gcoord =
                    (uint32_t)lcoord +
                    (gstate.currentCoordsInTile[0] * data.tile_size[0]);
                vec[current_filled + idx] = gcoord;
            }

        } else if (dest == 1) {
            // col column
            // fprintf(stderr, "d%lu\n", dest);
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t buf_idx = gstate.cell_idx + idx;
                auto lcoord = coords[1][buf_idx];
                auto gcoord =
                    (uint32_t)lcoord +
                    (gstate.currentCoordsInTile[1] * data.tile_size[1]);
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
                         offset, offsets[attrIdx], gstate.cell_idx);
        }
    }

    uint64_t produced = local_remains;
    gstate.cell_idx += produced;
    return produced;
}

uint64_t CsrReader::_PutDataNoPrune(optional_ptr<const FunctionData> bind_data,
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

    // the idxptrIdx of the current cell for the row idxptr
    uint32_t idxptrIdx = 0;
    for (uint64_t idx = 0; idx < data.tile_size[1]; idx++) {
        if (coords[0][idx + 1] >= gstate.cell_idx) {
            idxptrIdx = idx;
            break;
        }
    }

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t buf_idx = gstate.cell_idx + idx;

        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            auto colIdx = gstate.column_ids[i];

            // dimensions
            if (colIdx == 0) {
                // row column
                // adjust idxptrIdx
                while (coords[0][idxptrIdx + 1] <= buf_idx) {
                    idxptrIdx++;
                }

                // put the value
                auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
                auto lcoord = (uint32_t) idxptrIdx;
                auto gcoord =
                    (uint32_t)lcoord +
                    (gstate.currentCoordsInTile[0] * data.tile_size[0]);
                vec[current_filled + idx] = gcoord;
            } else if (colIdx == 1) {
                // col column
                auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
                auto lcoord = (uint32_t)coords[colIdx][buf_idx];
                auto gcoord =
                    (uint32_t)lcoord +
                    (gstate.currentCoordsInTile[1] * data.tile_size[1]);
                vec[current_filled + idx] = gcoord;
            } else {
                // attributes
                char *vals = (char *)pagevals;
                int attrIdx = colIdx - coords.size();
                auto type = data.attrTypes[attrIdx];
                CopyToVector(type, output, i, vals, current_filled,
                             local_remains, offset, offsets[attrIdx],
                             gstate.cell_idx);
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

uint64_t CsrReader::_PutDataNoPruneAndProjection(
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

    // the idxptrIdx of the current cell for the row idxptr
    uint32_t idxptrIdx = 0;
    for (uint64_t idx = 0; idx < data.tile_size[1]; idx++) {
        if (coords[0][idx + 1] >= gstate.cell_idx) {
            idxptrIdx = idx;
            break;
        }
    }

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t buf_idx = gstate.cell_idx + idx;

        // iterate over columns
        for (uint32_t i = 0; i < output.ColumnCount(); i++) {
            // dimensions
            if (i == 0) {
                // row column
                // adjust idxptrIdx
                while (coords[0][idxptrIdx + 1] <= buf_idx) {
                    idxptrIdx++;
                }

                // put the value
                auto vec = FlatVector::GetData<uint32_t>(output.data[0]);
                auto lcoord = (uint32_t) idxptrIdx;
                auto gcoord =
                    (uint32_t)lcoord +
                    (gstate.currentCoordsInTile[0] * data.tile_size[0]);
                vec[current_filled + idx] = gcoord;
            } else if (i == 1) {
                // col column
                auto vec = FlatVector::GetData<uint32_t>(output.data[1]);
                auto lcoord = (uint32_t)coords[i][buf_idx];
                auto gcoord =
                    (uint32_t)lcoord +
                    (gstate.currentCoordsInTile[1] * data.tile_size[1]);
                vec[current_filled + idx] = gcoord;
            } else {
                // attributes
                char *vals = (char *)pagevals;
                int attrIdx = i - coords.size();
                auto type = data.attrTypes[attrIdx];
                CopyToVector(type, output, i, vals, current_filled,
                             local_remains, offset, offsets[attrIdx], buf_idx);
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

uint64_t CsrReader::PutData(optional_ptr<const FunctionData> bind_data,
                            ArrayReadGlobalState &gstate, char *pagevals,
                            vector<uint64_t *> &coords, uint64_t num_of_cells,
                            DataChunk &output) {
    bool nullable = gstate.page->type == DENSE_FIXED_NULLABLE ||
                    gstate.page->type == SPARSE_FIXED_NULLABLE;

    if (nullable) {
        throw NotImplementedException("Nullable CSR is not supported yet");
    } else {
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
}
}  // namespace duckdb