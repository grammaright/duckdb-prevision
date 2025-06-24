
#include "csr_reader.hpp"

namespace duckdb {

void FillRowVector(uint32_t *dst, size_t dstOffset, uint64_t *src,
                   size_t srcCurrent, size_t cnt, vector<int> tileCoords,
                   vector<uint64_t> tileSize) {
  // first, find the idxptrIdx of the current cell
  uint32_t idxptrIdx = 0;
  for (uint64_t idx = 0; idx < tileSize[1]; idx++) {
    if (src[idx + 1] >= srcCurrent) {
      idxptrIdx = idx;
      break;
    }
  }

  // fill the output vector
  for (uint64_t idx = 0; idx < cnt; idx++) {
    uint32_t bufIdx = srcCurrent + idx;
    // adjust idxptrIdx
    while (src[idxptrIdx + 1] <= bufIdx) {
      idxptrIdx++;
    }
    // put the value
    auto lcoord = idxptrIdx;
    auto gcoord = (uint32_t)lcoord + (tileCoords[0] * tileSize[0]);
    dst[dstOffset + idx] = gcoord;
  }
}

uint64_t CsrReader::_PutData(optional_ptr<const FunctionData> bind_data,
                             ArrayReadGlobalState &gstate, char *pagevals,
                             vector<uint64_t *> &coords, uint64_t num_rows,
                             DataChunk &output) {
  auto &data = bind_data->Cast<ArrayReadData>();

  uint64_t current_filled = output.size();
  auto total_remains = num_rows - gstate.cell_idx;
  auto local_remains =
      std::min((uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

  // offsets for multi-attributes
  vector<idx_t> offsets = calculateOffsets(data.attrTypes, num_rows);

  // for each column
  for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
    auto dest = gstate.column_ids[gstate.projection_ids[i]];
    // dimensions
    if (dest == 0) {
      // row column
      auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
      FillRowVector(vec, current_filled, coords[0], gstate.cell_idx,
                    local_remains, gstate.currentCoordsInTile, data.tile_size);
    } else if (dest == 1) {
      // col column
      auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
      NCopyCoordsToVector(output.data[i], current_filled, coords[dest],
                          gstate.cell_idx, local_remains);
      // CopyToVector(LogicalType::UINTEGER, output, i, (char *)coords[dest],
      //              current_filled, local_remains, 0, gstate.cell_idx);
      UpdateToGlobalCoords(vec, current_filled, local_remains,
                           gstate.currentCoordsInTile[dest],
                           data.tile_size[dest]);
    } else {
      // attributes
      char *vals = (char *)pagevals;
      int attrIdx = dest - coords.size();
      auto type = data.attrTypes[attrIdx];

      NCopyToVector(output.data[i], current_filled, vals + offsets[attrIdx],
                    gstate.cell_idx, local_remains, type);
      // CopyToVector(type, output, i, vals, current_filled, local_remains,
      //            // ffsets[attrIdx], gstate.cell_idx);
    }
  }

  uint64_t produced = local_remains;
  gstate.cell_idx += produced;
  return produced;
}

uint64_t CsrReader::_PutDataNoPrune(optional_ptr<const FunctionData> bind_data,
                                    ArrayReadGlobalState &gstate,
                                    char *pagevals, vector<uint64_t *> &coords,
                                    uint64_t num_rows, DataChunk &output) {
  auto &data = bind_data->Cast<ArrayReadData>();

  uint64_t current_filled = output.size();
  auto total_remains = num_rows - gstate.cell_idx;
  auto local_remains =
      std::min((uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

  // offsets for multi-attributes
  vector<idx_t> offsets = calculateOffsets(data.attrTypes, num_rows);
  for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
    auto colIdx = gstate.column_ids[i];
    // dimensions
    if (colIdx == 0) {
      // row column
      auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
      FillRowVector(vec, current_filled, coords[0], gstate.cell_idx,
                    local_remains, gstate.currentCoordsInTile, data.tile_size);
    } else if (colIdx == 1) {
      // col column
      auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
      NCopyCoordsToVector(output.data[i], current_filled, coords[colIdx],
                          gstate.cell_idx, local_remains);
      UpdateToGlobalCoords(vec, current_filled, local_remains,
                           gstate.currentCoordsInTile[colIdx],
                           data.tile_size[colIdx]);
    } else {
      // attributes
      char *vals = (char *)pagevals;
      int attrIdx = colIdx - coords.size();
      auto type = data.attrTypes[attrIdx];

      NCopyToVector(output.data[i], current_filled, vals + offsets[attrIdx],
                    gstate.cell_idx, local_remains, type);
      // CopyToVector(type, output, i, vals, current_filled, local_remains,
      //              offsets[attrIdx], gstate.cell_idx);
    }
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
  auto local_remains =
      std::min((uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

  // offsets for multi-attributes
  vector<idx_t> offsets = calculateOffsets(data.attrTypes, num_rows);

  // iterate over columns
  for (uint32_t i = 0; i < output.ColumnCount(); i++) {
    // dimensions
    if (i == 0) {
      // row column
      auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
      FillRowVector(vec, current_filled, coords[0], gstate.cell_idx,
                    local_remains, gstate.currentCoordsInTile, data.tile_size);
    } else if (i == 1) {
      // col column
      auto vec = FlatVector::GetData<uint32_t>(output.data[1]);
      NCopyCoordsToVector(output.data[i], current_filled, coords[i],
                          gstate.cell_idx, local_remains);
      // CopyToVector(LogicalType::UINTEGER, output, i, (char *)coords[i],
      //              current_filled, local_remains, 0, gstate.cell_idx);
      UpdateToGlobalCoords(vec, current_filled, local_remains,
                           gstate.currentCoordsInTile[i], data.tile_size[i]);

    } else {
      // attributes
      char *vals = (char *)pagevals;
      int attrIdx = i - coords.size();
      auto type = data.attrTypes[attrIdx];

      NCopyToVector(output.data[i], current_filled, vals + offsets[attrIdx],
                    gstate.cell_idx, local_remains, type);
      // CopyToVector(type, output, i, vals, current_filled, local_remains,
      //              offsets[attrIdx], gstate.cell_idx);
    }
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
      return _PutDataNoPrune(bind_data, gstate, pagevals, coords, num_of_cells,
                             output);
    else  // projection_pushdown and filter_prune are both false
      return _PutDataNoPruneAndProjection(bind_data, gstate, pagevals, coords,
                                          num_of_cells, output);
  }
}
}  // namespace duckdb