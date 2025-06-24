
#include "coo_reader.hpp"

namespace duckdb {

template <typename T>
void debugFilledData(T *data, size_t cnt) {
  for (size_t i = 0; i < cnt; ++i) {
    std::cerr << data[i] << ",";
  }
  std::cerr << std::endl;
}

uint64_t CooReader::_PutData(optional_ptr<const FunctionData> bind_data,
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

  // std::cerr << "_PutData" << std::endl;

  // for each column
  for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
    auto dest = gstate.column_ids[gstate.projection_ids[i]];
    // dimensions
    if (dest < coords.size()) {
      auto vec = FlatVector::GetData<uint32_t>(output.data[i]);

      NCopyCoordsToVector(output.data[i], current_filled, coords[dest],
                          gstate.cell_idx, local_remains);
      // CopyToVector(LogicalType::UINTEGER, output, i, (char *)coords[dest],
      //              current_filled, local_remains, 0, gstate.cell_idx);
      UpdateToGlobalCoords(vec, current_filled, local_remains,
                           gstate.currentCoordsInTile[dest],
                           data.tile_size[dest]);

      // debugFilledData<uint64_t>(vec + current_filled, local_remains);
    } else {
      // attributes
      char *vals = (char *)pagevals;
      int attrIdx = dest - coords.size();
      auto type = data.attrTypes[attrIdx];

      NCopyToVector(output.data[i], current_filled, vals + offsets[attrIdx],
                    gstate.cell_idx, local_remains, type);
      // CopyToVector(type, output, i, vals, current_filled, local_remains,
      //              offsets[attrIdx], gstate.cell_idx);
      // auto vec = FlatVector::GetData<double>(output.data[i]);
      // debugFilledData<double>(vec + current_filled, local_remains);
    }
  }

  uint64_t produced = local_remains;
  gstate.cell_idx += produced;
  return produced;
}

uint64_t CooReader::_PutNullableData(optional_ptr<const FunctionData> bind_data,
                                     ArrayReadGlobalState &gstate,
                                     char *pagevals, vector<uint64_t *> &coords,
                                     uint64_t num_rows, DataChunk &output) {
  auto &data = bind_data->Cast<ArrayReadData>();

  // Calculating the maximum number of cells that can be filled
  uint64_t current_filled = output.size();
  auto total_remains = num_rows - gstate.cell_idx;
  auto local_remains =
      std::min((uint64_t)STANDARD_VECTOR_SIZE - current_filled, total_remains);

  // The number of actual produced cells
  uint64_t produced = 0;

  // offsets for multi-attributes
  vector<idx_t> offsets = calculateOffsets(data.attrTypes, num_rows);

  // for each column
  for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
    auto dest = gstate.column_ids[gstate.projection_ids[i]];
    // dimensions
    if (dest < coords.size()) {
      auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
      size_t filled = NCopyCoordsToVectorNullable(
          output.data[i], current_filled, coords[dest], gstate.cell_idx,
          local_remains, bf_util_get_nullbits(gstate.page));
      UpdateToGlobalCoords(vec, current_filled, filled,
                           gstate.currentCoordsInTile[dest],
                           data.tile_size[dest]);
      produced = filled;
    } else {
      // attributes
      char *vals = (char *)pagevals;
      int attrIdx = dest - coords.size();
      auto type = data.attrTypes[attrIdx];

      NCopyToVector(output.data[i], current_filled, vals + offsets[attrIdx],
                    gstate.cell_idx, local_remains, type);
      // CopyToVectorNullable(type, output, i, vals, current_filled,
      // local_remains, offsets[attrIdx], gstate.cell_idx, gstate.page);
    }
  }

  gstate.cell_idx += produced;
  return produced;
}

uint64_t CooReader::_PutDataNoPrune(optional_ptr<const FunctionData> bind_data,
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

  std::cerr << "[CooReader::_PutDataNoPrune] Is this called? If not, remove it "
               "and mark it as WIP."
            << std::endl;

  for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
    auto colIdx = gstate.column_ids[i];

    // dimensions
    if (colIdx < coords.size()) {
      auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
      NCopyCoordsToVector(output.data[i], current_filled, coords[colIdx],
                          gstate.cell_idx, local_remains);
      // CopyToVector(LogicalType::UINTEGER, output, i, (char *)coords[colIdx],
      //              current_filled, local_remains, 0, gstate.cell_idx);
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

uint64_t CooReader::_PutDataNoPruneAndProjection(
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

  std::cerr << "[CooReader::_PutDataNoPruneAndProjection] Is this called? If "
               "not, remove it and mark it as WIP."
            << std::endl;

  // iterate over columns
  for (uint32_t i = 0; i < output.ColumnCount(); i++) {
    // dimensions
    if (i < coords.size()) {
      auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
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

uint64_t CooReader::PutData(optional_ptr<const FunctionData> bind_data,
                            ArrayReadGlobalState &gstate, char *pagevals,
                            vector<uint64_t *> &coords, uint64_t num_of_cells,
                            DataChunk &output) {
  bool nullable = gstate.page->type == DENSE_FIXED_NULLABLE ||
                  gstate.page->type == SPARSE_FIXED_NULLABLE;
  if (nullable) {
    if (gstate.projection_ids.size() > 0) {  // filter_prune ON
      return _PutNullableData(bind_data, gstate, pagevals, coords, num_of_cells,
                              output);
    } else if (gstate.column_ids.size() ==
               output.data.size())  // no filter prune
      throw NotImplementedException(
          "_PutNullableDataNoPrune() is not supported yet");
    else  // projection_pushdown and filter_prune are both false
      throw NotImplementedException(
          "_PutNullableDataNoPruneAndProjection() is not supported yet");

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