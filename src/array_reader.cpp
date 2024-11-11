
#include "array_reader.hpp"

namespace duckdb {

uint64_t ArrayReader::_Put3DNullableData(optional_ptr<const FunctionData> bind_data,
                                 ArrayReadGlobalState &gstate, double *pagevals,
                                 uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();
    uint64_t max_processable = (uint64_t)STANDARD_VECTOR_SIZE;
    uint64_t num_processed = 0;

    vector<uint32_t> offsets;
    for (uint32_t i = 0; i < data.dim_len; i++) {
        offsets.push_back(gstate.currentCoordsInTile[i] * data.tile_size[i]);
    }

    uint64_t new_cell_idx = gstate.cell_idx;
    uint64_t current_filled = output.size();
    for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
        auto dest = gstate.column_ids[gstate.projection_ids[i]];
        if (dest == 0) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            uint64_t idx = current_filled;
            uint64_t buf_idx = gstate.cell_idx;
            while (idx < max_processable && buf_idx < size) {
                if (bf_util_is_cell_null(gstate.page, buf_idx)) {
                    buf_idx++;
                    continue;
                }
                    
                uint32_t coord =
                    buf_idx++ / (data.tile_size[1] * data.tile_size[2]);
                vec[idx++] = offsets[dest] + coord;
            }

            // set num_processed
            // note that it can seems duplicated, but it is necessary since
            // sometimes only specific columns are retrieved (e.g., SELECT
            // SUM(val) FROM read_array()).
            num_processed = idx - current_filled;
            new_cell_idx = buf_idx;
        } else if (dest == 1) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            uint64_t idx = current_filled;
            uint64_t buf_idx = gstate.cell_idx;
            while (idx < max_processable && buf_idx < size) {
                if (bf_util_is_cell_null(gstate.page, buf_idx)) {
                    buf_idx++;
                    continue;
                }

                uint32_t coord =
                    (buf_idx++ / data.tile_size[2]) % data.tile_size[1];
                vec[idx++] = offsets[dest] + coord;
            }

            num_processed = idx - current_filled;
            new_cell_idx = buf_idx;
        } else if (dest == 2) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            uint64_t idx = current_filled;
            uint64_t buf_idx = gstate.cell_idx;
            while (idx < max_processable && buf_idx < size) {
                if (bf_util_is_cell_null(gstate.page, buf_idx)) {
                    buf_idx++;
                    continue;
                }
                    
                uint32_t coord = buf_idx++ % data.tile_size[2];
                vec[idx++] = offsets[dest] + coord;
            }

            num_processed = idx - current_filled;
            new_cell_idx = buf_idx;
        } else if (dest == 3) {
            auto vec = FlatVector::GetData<double>(output.data[i]);
            uint64_t idx = current_filled;
            uint64_t buf_idx = gstate.cell_idx;
            while (idx < max_processable && buf_idx < size) {
                if (bf_util_is_cell_null(gstate.page, buf_idx)) {
                    buf_idx++;
                    continue;
                }
                    
                vec[idx++] = pagevals[buf_idx++];
            }

            num_processed = idx - current_filled;
            new_cell_idx = buf_idx;
        }
    }

    gstate.cell_idx = new_cell_idx;
    return num_processed;
}

uint64_t ArrayReader::_Put3DNullableDataNoPrune(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t max_processable = (uint64_t)STANDARD_VECTOR_SIZE;
    uint64_t current_filled = output.size();

    vector<uint32_t> offsets;
    for (uint32_t d = 0; d < data.dim_len; d++) {
        offsets.push_back(gstate.currentCoordsInTile[d] * data.tile_size[d]);
    }

    uint64_t idx = current_filled;
    uint64_t buf_idx = gstate.cell_idx;
    while (idx < max_processable && buf_idx < size) {
        if (bf_util_is_cell_null(gstate.page, buf_idx)) {
            buf_idx++;
            continue;
        }
            
        uint64_t *coords;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), data.dim_len,
            &coords);

        double val = pagevals[buf_idx];
        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            if (gstate.column_ids[i] == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] =
                    offsets[0] + coords[0];
            else if (gstate.column_ids[i] == 1)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] =
                    offsets[1] + coords[1];
            else if (gstate.column_ids[i] == 2)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] =
                    offsets[2] + coords[2];
            else if (gstate.column_ids[i] == 3)
                FlatVector::GetData<double>(output.data[i])[idx] = val;
        }

        buf_idx++;
        idx++;
    }

    gstate.cell_idx = buf_idx;
    return idx - current_filled;
}

uint64_t ArrayReader::_Put3DNullableDataNoPruneAndProjection(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t max_processable = (uint64_t)STANDARD_VECTOR_SIZE;
    uint64_t current_filled = output.size();

    vector<uint32_t> offsets;
    for (uint32_t d = 0; d < data.dim_len; d++) {
        offsets.push_back(gstate.currentCoordsInTile[d] * data.tile_size[d]);
    }

    auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
    auto ys = FlatVector::GetData<uint32_t>(output.data[1]);
    auto zs = FlatVector::GetData<uint32_t>(output.data[2]);
    auto vals =
        FlatVector::GetData<double>(output.data[3]);  // double type assumed

    uint64_t idx = current_filled;
    uint64_t buf_idx = gstate.cell_idx;
    while (idx < max_processable && buf_idx < size) {
        if (bf_util_is_cell_null(gstate.page, buf_idx)) {
            buf_idx++;
            continue;
        }
            
        uint64_t *coords;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), data.dim_len,
            &coords);
        double val = pagevals[buf_idx];
        xs[idx] = offsets[0] + coords[0];
        ys[idx] = offsets[1] + coords[1];
        zs[idx] = offsets[2] + coords[2];
        vals[idx] = val;

        buf_idx++;
        idx++;
    }

    gstate.cell_idx = buf_idx;
    return idx - current_filled;
}

uint64_t ArrayReader::_Put3DData(optional_ptr<const FunctionData> bind_data,
                                 ArrayReadGlobalState &gstate, double *pagevals,
                                 uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
        auto dest = gstate.column_ids[gstate.projection_ids[i]];
        uint32_t offset =
            gstate.currentCoordsInTile[dest] * data.tile_size[dest];
        if (dest == 0) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t vec_idx = current_filled + idx;
                uint32_t buf_idx = gstate.cell_idx + idx;
                uint32_t coord =
                    buf_idx / (data.tile_size[1] * data.tile_size[2]);
                vec[vec_idx] = offset + coord;
            }
        } else if (dest == 1) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t vec_idx = current_filled + idx;
                uint32_t buf_idx = gstate.cell_idx + idx;
                uint32_t coord =
                    (buf_idx / data.tile_size[2]) % data.tile_size[1];
                vec[vec_idx] = offset + coord;
            }
        } else if (dest == 2) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t vec_idx = current_filled + idx;
                uint32_t buf_idx = gstate.cell_idx + idx;
                uint32_t coord = buf_idx % data.tile_size[2];
                vec[vec_idx] = offset + coord;
            }
        } else if (dest == 3) {
            auto vec = FlatVector::GetData<double>(output.data[i]);
            memcpy(vec + current_filled, pagevals + gstate.cell_idx,
                   sizeof(double) * local_remains);
        }
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t ArrayReader::_Put3DDataNoPrune(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    vector<uint32_t> offsets;
    for (uint32_t d = 0; d < data.dim_len; d++) {
        offsets.push_back(gstate.currentCoordsInTile[d] * data.tile_size[d]);
    }

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t *coords;
        uint64_t buf_idx = gstate.cell_idx + idx;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), data.dim_len, &coords);
        double val = pagevals[buf_idx];

        auto vec_idx = current_filled + idx;
        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            if (gstate.column_ids[i] == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[vec_idx] =
                    offsets[0] + coords[0];
            else if (gstate.column_ids[i] == 1)
                FlatVector::GetData<uint32_t>(output.data[i])[vec_idx] =
                    offsets[1] + coords[1];
            else if (gstate.column_ids[i] == 2)
                FlatVector::GetData<uint32_t>(output.data[i])[vec_idx] =
                    offsets[2] + coords[2];
            else if (gstate.column_ids[i] == 3)
                FlatVector::GetData<double>(output.data[i])[vec_idx] = val;
        }
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t ArrayReader::_Put3DDataNoPruneAndProjection(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    vector<uint32_t> offsets;
    for (uint32_t d = 0; d < data.dim_len; d++) {
        offsets.push_back(gstate.currentCoordsInTile[d] * data.tile_size[d]);
    }

    auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
    auto ys = FlatVector::GetData<uint32_t>(output.data[1]);
    auto zs = FlatVector::GetData<uint32_t>(output.data[2]);
    auto vals =
        FlatVector::GetData<double>(output.data[3]);  // double type assumed

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t *coords;
        uint64_t buf_idx = gstate.cell_idx + idx;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), data.dim_len, &coords);

        double val = pagevals[buf_idx];
        auto vec_idx = current_filled + idx;

        xs[vec_idx] = offsets[0] + coords[0];
        ys[vec_idx] = offsets[1] + coords[1];
        zs[vec_idx] = offsets[2] + coords[2];
        vals[vec_idx] = val;

        // std::cout << "\t[Put2DDataNoPruneAndProjection] idx: " << idx <<
        // ", x: " << xs[idx] << ", y: " << ys[idx] << ", val: " <<
        // vals[idx] << std::endl; free(coords);
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t ArrayReader::_Put2DData(optional_ptr<const FunctionData> bind_data,
                                 ArrayReadGlobalState &gstate, double *pagevals,
                                 uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    auto total_remains = size - gstate.cell_idx;
    auto local_remains = std::min(
        (uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    // for dest == 1 and 2
    for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
        auto dest = gstate.column_ids[gstate.projection_ids[i]];
        uint32_t offset =
            gstate.currentCoordsInTile[dest] * data.tile_size[dest];
        if (dest == 0) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t vec_idx = current_filled + idx;
                uint32_t buf_idx = gstate.cell_idx + idx;
                uint32_t coord = buf_idx / data.tile_size[1];
                vec[vec_idx] = offset + coord;
            }
        } else if (dest == 1) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t vec_idx = current_filled + idx;
                uint32_t buf_idx = gstate.cell_idx + idx;
                uint32_t coord = buf_idx % data.tile_size[1];
                vec[vec_idx] = offset + coord;
            }
        } else if (dest == 2) {
            auto vec = FlatVector::GetData<double>(output.data[i]);
            memcpy(vec + current_filled, pagevals + gstate.cell_idx,
                   sizeof(double) * local_remains);
        }
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t ArrayReader::_Put2DDataNoPrune(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    auto total_remains = size - gstate.cell_idx;
    auto local_remains = std::min(
        (uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);
    
    vector<uint32_t> offsets;
    for (uint32_t d = 0; d < data.dim_len; d++) {
        offsets.push_back(gstate.currentCoordsInTile[d] * data.tile_size[d]);
    }

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t *coords;
        uint64_t buf_idx = gstate.cell_idx + idx;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), 2, &coords);

        double val = pagevals[buf_idx];

        uint32_t vec_idx = current_filled + idx;
        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            if (gstate.column_ids[i] == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[vec_idx] =
                    offsets[0] + coords[0];
            else if (gstate.column_ids[i] == 1)
                FlatVector::GetData<uint32_t>(output.data[i])[vec_idx] =
                    offsets[1] + coords[1];
            else if (gstate.column_ids[i] == 2)
                FlatVector::GetData<double>(output.data[i])[vec_idx] = val;
        }

        // std::cout << "\t[Put2DDataNoPrune] idx: " << idx << ", x: " <<
        // coords[0] << ", y: " << coords[1] << ", val: " << val <<
        // std::endl; free(coords);
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t ArrayReader::_Put2DDataNoPruneAndProjection(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    auto total_remains = size - gstate.cell_idx;
    auto local_remains = std::min(
        (uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);
    
    vector<uint32_t> offsets;
    for (uint32_t d = 0; d < data.dim_len; d++) {
        offsets.push_back(gstate.currentCoordsInTile[d] * data.tile_size[d]);
    }

    auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
    auto ys = FlatVector::GetData<uint32_t>(output.data[1]);
    auto vals =
        FlatVector::GetData<double>(output.data[2]);  // double type assumed

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t *coords;
        uint64_t buf_idx = gstate.cell_idx + idx;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), 2, &coords);

        double val = pagevals[buf_idx];
        uint32_t vec_idx = current_filled + idx;

        xs[vec_idx] = offsets[0] + coords[0];
        ys[vec_idx] = offsets[1] + coords[1];
        vals[vec_idx] = val;

        // std::cout << "\t[Put2DDataNoPruneAndProjection] idx: " << idx <<
        // ", x: " << xs[idx] << ", y: " << ys[idx] << ", val: " <<
        // vals[idx] << std::endl; free(coords);
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t ArrayReader::_Put2DNullableData(optional_ptr<const FunctionData> bind_data,
                                 ArrayReadGlobalState &gstate, double *pagevals,
                                 uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();
    uint64_t current_filled = output.size();
    uint64_t max_processable = (uint64_t)STANDARD_VECTOR_SIZE;

    vector<uint32_t> offsets;
    for (uint32_t d = 0; d < data.dim_len; d++) {
        offsets.push_back(gstate.currentCoordsInTile[d] * data.tile_size[d]);
    }

    uint64_t num_processed = 0;
    uint64_t new_cell_idx = gstate.cell_idx;

    for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
        auto dest = gstate.column_ids[gstate.projection_ids[i]];
        if (dest == 0) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            uint64_t idx = current_filled;
            uint64_t buf_idx = gstate.cell_idx;
            while (idx < max_processable && buf_idx < size) {
                if (bf_util_is_cell_null(gstate.page, buf_idx)) {
                    buf_idx++;
                    continue;
                }
                    
                uint32_t coord =
                    buf_idx++ / data.tile_size[1];
                vec[idx++] = offsets[dest] + coord;
            }

            // set num_processed only here
            // note that it can seems duplicated, but it is necessary since
            // sometimes only specific columns are retrieved (e.g., SELECT
            // * FROM array)
            num_processed = idx - current_filled;
            new_cell_idx = buf_idx;
        } else if (dest == 1) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            uint64_t idx = current_filled;
            uint64_t buf_idx = gstate.cell_idx;
            while (idx < max_processable && buf_idx < size) {
                if (bf_util_is_cell_null(gstate.page, buf_idx)) {
                    buf_idx++;
                    continue;
                }
                    
                uint32_t coord = buf_idx++ % data.tile_size[1];
                vec[idx++] = offsets[dest] + coord;
            }

            num_processed = idx - current_filled;
            new_cell_idx = buf_idx;
        } else if (dest == 2) {
            auto vec = FlatVector::GetData<double>(output.data[i]);
            uint64_t idx = current_filled;
            uint64_t buf_idx = gstate.cell_idx;
            while (idx < max_processable && buf_idx < size) {
                if (bf_util_is_cell_null(gstate.page, buf_idx)) {
                    buf_idx++;
                    continue;
                }
                    
                vec[idx++] = pagevals[buf_idx++];
            }

            num_processed = idx - current_filled;
            new_cell_idx = offsets[dest] + buf_idx;
        }
    }

    gstate.cell_idx = new_cell_idx;
    return num_processed;
}

uint64_t ArrayReader::_Put2DNullableDataNoPrune(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    uint64_t max_processable = (uint64_t)STANDARD_VECTOR_SIZE;

    vector<uint32_t> offsets;
    for (uint32_t d = 0; d < data.dim_len; d++) {
        offsets.push_back(gstate.currentCoordsInTile[d] * data.tile_size[d]);
    }

    uint64_t idx = current_filled;
    uint64_t buf_idx = gstate.cell_idx;
    while (idx < max_processable && buf_idx < size) {
        if (bf_util_is_cell_null(gstate.page, buf_idx)) {
            buf_idx++;
            continue;
        }
            
        uint64_t *coords;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), data.dim_len,
            &coords);

        double val = pagevals[buf_idx];
        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            if (gstate.column_ids[i] == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] =
                    offsets[0] + coords[0];
            else if (gstate.column_ids[i] == 1)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] =
                    offsets[1] + coords[1];
            else if (gstate.column_ids[i] == 2)
                FlatVector::GetData<double>(output.data[i])[idx] = val;
        }

        buf_idx++;
        idx++;
    }

    gstate.cell_idx = buf_idx;
    return idx - current_filled;
}

uint64_t ArrayReader::_Put2DNullableDataNoPruneAndProjection(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    uint64_t max_processable = (uint64_t)STANDARD_VECTOR_SIZE;

    vector<uint32_t> offsets;
    for (uint32_t d = 0; d < data.dim_len; d++) {
        offsets.push_back(gstate.currentCoordsInTile[d] * data.tile_size[d]);
    }

    auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
    auto ys = FlatVector::GetData<uint32_t>(output.data[1]);
    auto vals =
        FlatVector::GetData<double>(output.data[2]);  // double type assumed

    uint64_t idx = current_filled;
    uint64_t buf_idx = gstate.cell_idx;
    while (idx < max_processable && buf_idx < size) {
        if (bf_util_is_cell_null(gstate.page, buf_idx)) {
            buf_idx++;
            continue;
        }
            
        uint64_t *coords;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), data.dim_len,
            &coords);
        double val = pagevals[buf_idx];
        xs[idx] = offsets[0] + coords[0];
        ys[idx] = offsets[1] + coords[1];
        vals[idx] = val;

        buf_idx++;
        idx++;
    }

    gstate.cell_idx = buf_idx;
    return idx - current_filled;
}

uint64_t ArrayReader::_Put1DData(optional_ptr<const FunctionData> bind_data,
                                 ArrayReadGlobalState &gstate, double *pagevals,
                                 uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    uint64_t current_filled = output.size();
    uint32_t offset = gstate.currentCoordsInTile[0] * data.tile_size[0];

    auto total_remains = size - gstate.cell_idx;
    auto local_remains = std::min(
        (uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        auto buf_idx = gstate.cell_idx + idx;
        double val = pagevals[buf_idx];

        uint64_t vec_idx = current_filled + idx;
        for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
            auto dest = gstate.column_ids[gstate.projection_ids[i]];
            if (dest == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[vec_idx] =
                    offset + buf_idx;
            else if (dest == 1)
                FlatVector::GetData<double>(output.data[i])[vec_idx] = val;
        }

        // std::cout << "\t[Put1DData] idx: " << idx << ", val: " << val <<
        // std::endl;
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t ArrayReader::_Put1DDataNoPrune(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();
    uint64_t current_filled = output.size();
    uint32_t offset = gstate.currentCoordsInTile[0] * data.tile_size[0];

    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        auto buf_idx = gstate.cell_idx + idx;
        double val = pagevals[buf_idx];

        uint64_t vec_idx = current_filled + idx;
        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            if (gstate.column_ids[i] == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[vec_idx] =
                    offset + buf_idx;
            else if (gstate.column_ids[i] == 1)
                FlatVector::GetData<double>(output.data[i])[vec_idx] = val;
        }

        // std::cout << "\t[Put1DDataNoPrune] idx: " << idx << ", val: " <<
        // val
        // << std::endl;
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t ArrayReader::_Put1DDataNoPruneAndProjection(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();
    uint32_t offset = gstate.currentCoordsInTile[0] * data.tile_size[0];

    uint64_t current_filled = output.size();
    auto total_remains = size - gstate.cell_idx;
    auto local_remains = std::min(
        (uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

    auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
    auto vals =
        FlatVector::GetData<double>(output.data[1]);  // double type assumed

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t *coords;
        uint64_t buf_idx = gstate.cell_idx + idx;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), 2, &coords);

        double val = pagevals[buf_idx];
        uint64_t vec_idx = current_filled + idx;

        xs[vec_idx] = offset + buf_idx;
        vals[vec_idx] = val;

        // std::cout << "\t[Put1DDataNoPruneAndProjection] idx: " << idx <<
        // ", val: " << vals[idx] << std::endl;

        // free(coords);
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t ArrayReader::PutData(optional_ptr<const FunctionData> bind_data,
                              ArrayReadGlobalState &gstate, double *pagevals,
                              uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();
    bool nullable = gstate.page->type == DENSE_FIXED_NULLABLE ||
                    gstate.page->type == SPARSE_FIXED_NULLABLE;

    // TODO: dimension-free code?
    if (data.dim_len == 3) {
        if (nullable) {
            if (gstate.projection_ids.size() > 0)  // filter_prune ON
                return _Put3DNullableData(bind_data, gstate, pagevals, size, output);
            else if (gstate.column_ids.size() ==
                    output.data.size())  // no filter prune
                return _Put3DNullableDataNoPrune(bind_data, gstate, pagevals, size, output);
            else  // projection_pushdown and filter_prune are both false
                return _Put3DNullableDataNoPruneAndProjection(bind_data, gstate, pagevals,
                                                    size, output);
        } else {
            if (gstate.projection_ids.size() > 0)  // filter_prune ON
                return _Put3DData(bind_data, gstate, pagevals, size, output);
            else if (gstate.column_ids.size() ==
                    output.data.size())  // no filter prune
                return _Put3DDataNoPrune(bind_data, gstate, pagevals, size, output);
            else  // projection_pushdown and filter_prune are both false
                return _Put3DDataNoPruneAndProjection(bind_data, gstate, pagevals,
                                                    size, output);
        }
    } else if (data.dim_len == 2) {
        if (nullable) {
            if (gstate.projection_ids.size() > 0)  // filter_prune ON
                return _Put2DNullableData(bind_data, gstate, pagevals, size, output);
            else if (gstate.column_ids.size() ==
                    output.data.size())  // no filter prune
                return _Put2DNullableDataNoPrune(bind_data, gstate, pagevals, size, output);
            else  // projection_pushdown and filter_prune are both false
                return _Put2DNullableDataNoPruneAndProjection(bind_data, gstate, pagevals,
                                                    size, output);
        } else {
            if (gstate.projection_ids.size() > 0)  // filter_prune ON
                return _Put2DData(bind_data, gstate, pagevals, size, output);
            else if (gstate.column_ids.size() ==
                    output.data.size())  // no filter prune
                return _Put2DDataNoPrune(bind_data, gstate, pagevals, size, output);
            else  // projection_pushdown and filter_prune are both false
                return _Put2DDataNoPruneAndProjection(bind_data, gstate, pagevals,
                                                    size, output);
        }
    } else {
        if (nullable) {
            throw std::runtime_error("Nullable array is not supported currently");
        }

        if (gstate.projection_ids.size() > 0)  // filter_prune ON
            return _Put1DData(bind_data, gstate, pagevals, size, output);
        else if (gstate.column_ids.size() ==
                 output.data.size())  // no filter prune
            return _Put1DDataNoPrune(bind_data, gstate, pagevals, size, output);
        else  // projection_pushdown and filter_prune are both false
            return _Put1DDataNoPruneAndProjection(bind_data, gstate, pagevals,
                                                  size, output);
    }
}
}  // namespace duckdb