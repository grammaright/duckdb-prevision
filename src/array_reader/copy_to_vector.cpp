
#include "coo_reader.hpp"

namespace duckdb {

void CopyToVectorNullable(LogicalType type,   // the type of the attribute
                          DataChunk &output,  // the output data chunk
                          idx_t vecIdx,       // the index of the vector in the data chunk
                          char *vals,         // the array data
                          uint64_t current_filled,  // the current number of filled cells in the vector
                          uint64_t local_remains,   // the maximum number of cells to fill
                          idx_t dataLen,            // the length of the cell
                          idx_t offset,             // the offset of the cell
                          idx_t cell_starting_idx,  // the starting index of the cell
                          PFpage* page) {           // the page
    uint64_t filled = 0;
    // TODO: support more types
    if (type == LogicalType::INTEGER) {
        auto vec = FlatVector::GetData<int32_t>(output.data[vecIdx]);
        for (uint64_t idx = 0; idx < local_remains; idx++) {
            uint64_t buf_idx = cell_starting_idx + idx;
            // continue if the cell is null
            if (bf_util_is_cell_null(page, buf_idx)) 
                continue;

            vec[current_filled + filled++] =
                *(int32_t *)(vals + (buf_idx * dataLen) + offset);
        }
    } else if (type == LogicalType::FLOAT) {
        auto vec = FlatVector::GetData<float>(output.data[vecIdx]);
        for (uint64_t idx = 0; idx < local_remains; idx++) {
            uint64_t buf_idx = cell_starting_idx + idx;
            // continue if the cell is null
            if (bf_util_is_cell_null(page, buf_idx)) 
                continue;

            vec[current_filled + filled++] =
                *(float *)(vals + (buf_idx * dataLen) + offset);
        }
    } else if (type == LogicalType::DOUBLE) {
        auto vec = FlatVector::GetData<double>(output.data[vecIdx]);
        for (uint64_t idx = 0; idx < local_remains; idx++) {
            uint64_t buf_idx = cell_starting_idx + idx;
            // continue if the cell is null
            if (bf_util_is_cell_null(page, buf_idx)) 
                continue;

            vec[current_filled + filled++] =
                *(double *)(vals + (buf_idx * dataLen) + offset);
            // fprintf(stderr, "\tvec[%ld] = %lf\n", idx, vec[idx]);
        }
    } else {
        throw NotImplementedException("Unsupported type");
    }
}

void CopyToVector(LogicalType type,     // the type of the attribute
                  DataChunk &output,    // the output data chunk
                  idx_t vecIdx,         // the index of the vector in the data chunk
                  char *vals,           // the array data
                  uint64_t current_filled,  // the current number of filled cells in the vector
                  uint64_t local_remains,   // the maximum number of cells to fill
                  idx_t dataLen,            // the length of the cell
                  idx_t offset,           // the offset of the cell
                  idx_t cell_starting_idx) {  // the starting index of the cell

    // TODO: support more types
    if (type == LogicalType::INTEGER) {
        auto vec = FlatVector::GetData<int32_t>(output.data[vecIdx]);
        for (uint64_t idx = 0; idx < local_remains; idx++) {
            uint64_t buf_idx = cell_starting_idx + idx;
            vec[current_filled + idx] =
                *(int32_t *)(vals + (buf_idx * dataLen) + offset);
            // fprintf(stderr, "\tvec[%ld] = %d\n", idx, vec[idx]);
        }
    } else if (type == LogicalType::FLOAT) {
        auto vec = FlatVector::GetData<float>(output.data[vecIdx]);
        for (uint64_t idx = 0; idx < local_remains; idx++) {
            uint64_t buf_idx = cell_starting_idx + idx;
            vec[current_filled + idx] =
                *(float *)(vals + (buf_idx * dataLen) + offset);
        }
    } else if (type == LogicalType::DOUBLE) {
        auto vec = FlatVector::GetData<double>(output.data[vecIdx]);
        for (uint64_t idx = 0; idx < local_remains; idx++) {
            uint64_t buf_idx = cell_starting_idx + idx;
            vec[current_filled + idx] =
                *(double *)(vals + (buf_idx * dataLen) + offset);
            // fprintf(stderr, "\tvec[%ld] = %lf\n", idx, vec[idx]);
        }
    } else {
        throw NotImplementedException("Unsupported type");
    }
}

}  // namespace duckdb