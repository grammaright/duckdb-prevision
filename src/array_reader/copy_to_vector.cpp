
#include "coo_reader.hpp"

namespace duckdb {

void CopyToVectorNullable(
    LogicalType type,   // the type of the attribute
    DataChunk &output,  // the output data chunk
    idx_t vecIdx,       // the index of the vector in the data chunk
    char *vals,         // the array data
    uint64_t
        current_filled,      // the current number of filled cells in the vector
    uint64_t local_remains,  // the maximum number of cells to fill
    idx_t offset,            // the offset of the current attribute
    idx_t cell_starting_idx,  // the starting index of the cell
    PFpage *page) {           // the page
  uint64_t filled = 0;
  // TODO: support more types
  if (type == LogicalType::INTEGER) {
    auto vec = FlatVector::GetData<int32_t>(output.data[vecIdx]);
    for (uint64_t idx = 0; idx < local_remains; idx++) {
      uint64_t buf_idx = cell_starting_idx + idx;
      // continue if the cell is null
      if (bf_util_is_cell_null(page, buf_idx)) continue;

      vec[current_filled + filled++] = *((int32_t *)vals + offset + buf_idx);
    }
  } else if (type == LogicalType::FLOAT) {
    auto vec = FlatVector::GetData<float>(output.data[vecIdx]);
    for (uint64_t idx = 0; idx < local_remains; idx++) {
      uint64_t buf_idx = cell_starting_idx + idx;
      // continue if the cell is null
      if (bf_util_is_cell_null(page, buf_idx)) continue;

      vec[current_filled + filled++] = *((float *)vals + offset + buf_idx);
    }
  } else if (type == LogicalType::DOUBLE) {
    auto vec = FlatVector::GetData<double>(output.data[vecIdx]);
    for (uint64_t idx = 0; idx < local_remains; idx++) {
      uint64_t buf_idx = cell_starting_idx + idx;
      // continue if the cell is null
      if (bf_util_is_cell_null(page, buf_idx)) continue;

      vec[current_filled + filled++] = *((double *)vals + offset + buf_idx);
    }
  } else {
    throw NotImplementedException("Unsupported type");
  }
}

void CopyToVector(
    LogicalType type,   // the type of the attribute
    DataChunk &output,  // the output data chunk
    idx_t vecIdx,       // the index of the vector in the data chunk
    char *vals,         // the array data
    uint64_t
        current_filled,      // the current number of filled cells in the vector
    uint64_t local_remains,  // the maximum number of cells to fill
    idx_t offset,            // the offset of the current attribute
    idx_t cell_starting_idx) {  // the starting index of the cell

  // TODO: support more types
  if (type == LogicalType::INTEGER) {
    auto vec = FlatVector::GetData<int32_t>(output.data[vecIdx]);
    memcpy(vec + current_filled, (int32_t *)vals + offset + cell_starting_idx,
           local_remains * sizeof(int32_t));
  } else if (type == LogicalType::FLOAT) {
    auto vec = FlatVector::GetData<float>(output.data[vecIdx]);
    memcpy(vec + current_filled, (float *)vals + offset + cell_starting_idx,
           local_remains * sizeof(float));
  } else if (type == LogicalType::DOUBLE) {
    auto vec = FlatVector::GetData<double>(output.data[vecIdx]);
    memcpy(vec + current_filled, (double *)vals + offset + cell_starting_idx,
           local_remains * sizeof(double));
  } else {
    throw NotImplementedException("Unsupported type");
  }
}

size_t NCopyToVectorNullable(Vector &dst, size_t dstOffset, void *src,
                             size_t srcOffset, size_t cnt, LogicalType type,
                             uint8_t *nullbits) {
  size_t filled = 0;

  // TODO: support more types
  if (type == LogicalType::INTEGER) {
    auto vec = FlatVector::GetData<int32_t>(dst);
    for (uint64_t idx = 0; idx < cnt; idx++) {
      if (bf_util_is_cell_null(nullbits, srcOffset + idx)) continue;
      vec[dstOffset + filled++] = *((int32_t *)src + srcOffset + idx);
    }
  } else if (type == LogicalType::UINTEGER) {
    auto vec = FlatVector::GetData<uint32_t>(dst);
    for (uint64_t idx = 0; idx < cnt; idx++) {
      if (bf_util_is_cell_null(nullbits, srcOffset + idx)) continue;
      vec[dstOffset + filled++] = *((uint32_t *)src + srcOffset + idx);
    }
  } else if (type == LogicalType::UBIGINT) {
    auto vec = FlatVector::GetData<uint64_t>(dst);
    for (uint64_t idx = 0; idx < cnt; idx++) {
      if (bf_util_is_cell_null(nullbits, srcOffset + idx)) continue;
      vec[dstOffset + filled++] = *((uint64_t *)src + srcOffset + idx);
    }
  } else if (type == LogicalType::FLOAT) {
    auto vec = FlatVector::GetData<float>(dst);
    for (uint64_t idx = 0; idx < cnt; idx++) {
      if (bf_util_is_cell_null(nullbits, srcOffset + idx)) continue;
      vec[dstOffset + filled++] = *((float *)src + srcOffset + idx);
    }
  } else if (type == LogicalType::DOUBLE) {
    auto vec = FlatVector::GetData<double>(dst);
    for (uint64_t idx = 0; idx < cnt; idx++) {
      if (bf_util_is_cell_null(nullbits, srcOffset + idx)) continue;
      vec[dstOffset + filled++] = *((double *)src + srcOffset + idx);
    }
  } else {
    throw NotImplementedException("Unsupported type");
  }

  return filled;
}

size_t NCopyCoordsToVectorNullable(Vector &dst, size_t dstOffset, void *src,
                                   size_t srcOffset, size_t cnt,
                                   uint8_t *nullbits) {
  size_t filled = 0;
  auto vec = FlatVector::GetData<uint32_t>(dst);
  for (uint64_t idx = 0; idx < cnt; idx++) {
    if (bf_util_is_cell_null(nullbits, srcOffset + idx)) continue;
    vec[dstOffset + filled++] = (uint32_t)*((uint64_t *)src + srcOffset + idx);
  }

  return filled;
}

void NCopyCoordsToVector(Vector &dst, size_t dstOffset, void *src,
                         size_t srcOffset, size_t cnt) {
  auto vec = FlatVector::GetData<uint32_t>(dst);
  for (uint64_t idx = 0; idx < cnt; idx++) {
    vec[dstOffset + idx] = (uint32_t)*((uint64_t *)src + srcOffset + idx);
  }
}

void NCopyToVector(Vector &dst, size_t dstOffset, void *src, size_t srcOffset,
                   size_t cnt, LogicalType type) {
  // TODO: support more types
  auto typeSize = GetTypeIdSize(type.InternalType());
  if (type == LogicalType::INTEGER) {
    auto vec = FlatVector::GetData<int32_t>(dst);
    memcpy(vec + dstOffset, (int32_t *)src + srcOffset, cnt * typeSize);
  } else if (type == LogicalType::UINTEGER) {
    auto vec = FlatVector::GetData<uint32_t>(dst);
    memcpy(vec + dstOffset, (uint32_t *)src + srcOffset, cnt * typeSize);
  } else if (type == LogicalType::UBIGINT) {
    auto vec = FlatVector::GetData<uint64_t>(dst);
    memcpy(vec + dstOffset, (uint64_t *)src + srcOffset, cnt * typeSize);
  } else if (type == LogicalType::FLOAT) {
    auto vec = FlatVector::GetData<float>(dst);
    memcpy(vec + dstOffset, (float *)src + srcOffset, cnt * typeSize);
  } else if (type == LogicalType::DOUBLE) {
    auto vec = FlatVector::GetData<double>(dst);
    memcpy(vec + dstOffset, (double *)src + srcOffset, cnt * typeSize);
  } else {
    throw NotImplementedException("Unsupported type");
  }
}

vector<idx_t> calculateOffsets(const vector<LogicalType> &attrTypes,
                               uint64_t num_rows) {
  vector<idx_t> offsets;
  idx_t offset = 0;  // final value of it will be a size of a row
  for (uint32_t i = 0; i < attrTypes.size(); i++) {
    offsets.push_back(offset);
    offset += GetTypeIdSize(attrTypes[i].InternalType()) * num_rows;
  }
  return offsets;
}

void UpdateToGlobalCoords(uint32_t *dst, uint64_t dstOffset, uint64_t cnt,
                          int tileCoordInDim, int tileSizeInDim) {
  // update to global coordinates
  auto gcoordOffset = tileCoordInDim * tileSizeInDim;
  for (uint64_t idx = 0; idx < cnt; idx++) {
    dst[dstOffset + idx] += gcoordOffset;
  }
}

}  // namespace duckdb