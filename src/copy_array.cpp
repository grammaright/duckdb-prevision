#include "copy_array.hpp"

#include "array_extension.hpp"

#include "buffer/bf.h"

namespace duckdb {
COOToArrayWriteArrayData::COOToArrayWriteArrayData(string file_path)
    : WriteArrayData(file_path) {}

DenseToTileWriteArrayData::DenseToTileWriteArrayData(
    string file_path, vector<uint64_t> _tile_coords)
    : WriteArrayData(file_path), tile_coords(_tile_coords) {}

WriteArrayData::WriteArrayData(string file_path) {
    array_name = file_path.substr(0, file_path.find_last_of("."));

    uint64_t **_dim_domains;
    uint64_t *_tile_size;
    uint64_t *_array_size_in_tile;
    storage_util_get_dim_domains(array_name.c_str(), &_dim_domains, &dim_len);
    storage_util_get_tile_extents(array_name.c_str(), &_tile_size, &dim_len);
    storage_util_get_dcoord_lens(_dim_domains, _tile_size, dim_len,
                                 &_array_size_in_tile);

    array_size_in_tile =
        vector<uint64_t>(_array_size_in_tile, _array_size_in_tile + dim_len);
    tile_size = vector<uint64_t>(_tile_size, _tile_size + dim_len);

    storage_util_free_dim_domains(&_dim_domains, dim_len);
    storage_util_free_tile_extents(&_tile_size, dim_len);
    storage_util_free_dcoord_lens(&_array_size_in_tile);
}

unique_ptr<FunctionData> WriteArrayData::Copy() const { return nullptr; }

bool WriteArrayData::Equals(const FunctionData &other_p) const { return false; }

GlobalWriteArrayData::GlobalWriteArrayData(ClientContext &context,
                                           FunctionData &bind_data,
                                           const string &file_path) {
    auto &data = bind_data.Cast<WriteArrayData>();

    auto arrname = data.array_name.c_str();
    arrname_char = new char[1024];
    strcpy(arrname_char, arrname);

    dim_len = data.dim_len;
    tile_coords = new uint64_t[dim_len];

    is_pinned = false;

    // if data is instance of COOToArrayWriteArrayData
    if (dynamic_cast<DenseToTileWriteArrayData *>(&data)) {
        auto &array_data = data.Cast<DenseToTileWriteArrayData>();
        writer = make_uniq<DenseToTileCopyArrayWriter>(*this, array_data);
    } else if (dynamic_cast<COOToArrayWriteArrayData *>(&data)) {
        writer = make_uniq<COOToArrayCopyArrayWriter>();
    }

    // disable preemptive eviction for future array access (e.g., another table
    // or array queries)
    BF_DisablePE(arrname_char);

    ArrayExtension::ResetPVBufferStats();
}

DenseToTileCopyArrayWriter::DenseToTileCopyArrayWriter(
    GlobalFunctionData &gstate, DenseToTileWriteArrayData &array_data) {
    auto &array_gstate = gstate.Cast<GlobalWriteArrayData>();

    array_gstate.pin(array_data.tile_coords);

    cur_idx = 0;
}

COOToArrayCopyArrayWriter::COOToArrayCopyArrayWriter() {}

void GlobalWriteArrayData::pin(vector<uint64_t> _tile_coords) {
    assert(!is_pinned);

    // getbuffer
    for (uint32_t i = 0; i < this->dim_len; i++) {
        tile_coords[i] = _tile_coords[i];
    }

    // TODO: Consider sparse tile in the future
    key = {arrname_char, tile_coords, dim_len, BF_EMPTYTILE_DENSE};

    int res = BF_GetBuf(key, &page);
    assert(res == BFE_OK);

    buf_size = page->pagebuf_len / sizeof(double);
    buf = (double *)bf_util_get_pagebuf(page);

    // nullbits set to 0 if it is the first time to pin the tile  
    if (page->type == DENSE_FIXED_NULLABLE 
            || page->type == SPARSE_FIXED_NULLABLE) {
        bool is_first = pinned_tiles.find(_tile_coords) == pinned_tiles.end();
        if (is_first) {
            bf_util_set_nullbits_null(page);

            // push to pinned tiles
            pinned_tiles.insert(_tile_coords);
        } 
    }

    is_pinned = true;
}

void GlobalWriteArrayData::unpin() {
    if (!is_pinned) {
        return;
    }

    BF_TouchBuf(key);
    BF_UnpinBuf(key);

    page = NULL;
    is_pinned = false;
}

// vector<uint32_t> GlobalWriteArrayData::GetTileCoords() {
//     return vector<uint32_t>(tile_coords, tile_coords + dim_len);
// }

static unique_ptr<FunctionData> WriteArrayBind(
    ClientContext &context, CopyFunctionBindInput &input,
    const vector<string> &names, const vector<LogicalType> &sql_types) {
    ArrayCopyFunctionExecutionMode mode = COO_TO_ARRAY;
    uint64_t x, y, z;
    uint32_t dim_len = 0;

    // check all the options in the copy info
    for (auto &option : input.info.options) {
        if (option.first == "COORD_X") {
            auto incoords = option.second;
            for (auto incoord : incoords) {
                auto val = incoord.GetValue<uint64_t>();
                x = val;
                dim_len = (dim_len > 1) ? dim_len : 1;
                break;  // I don't know why it gives a vector
            }
        } else if (option.first == "COORD_Y") {
            auto incoords = option.second;
            for (auto incoord : incoords) {
                auto val = incoord.GetValue<uint64_t>();
                y = val;
                dim_len = (dim_len > 2) ? dim_len : 2;
                break;  // I don't know why it gives a vector
            }
        } else if (option.first == "COORD_Z") {
            auto incoords = option.second;
            for (auto incoord : incoords) {
                auto val = incoord.GetValue<uint64_t>();
                z = val;
                dim_len = (dim_len > 3) ? dim_len : 3;
                break;  // I don't know why it gives a vector
            }
        } else if (option.first == "MODE") {
            auto mode_code = option.second[0].GetValue<int>();
            if (mode_code == 0) {
                mode = COO_TO_ARRAY;
            } else if (mode_code == 1) {
                mode = VALUES_TO_TILE;
            } else {
                throw NotImplementedException("Unknown mode for WriteArray: %d",
                                              mode_code);
            }
        } else {
            throw NotImplementedException("Unknown option for WriteArray: %s",
                                          option.first.c_str());
        }
    }

    if (mode == COO_TO_ARRAY) {
        auto file_path = input.info.file_path;
        auto bind_data = make_uniq<COOToArrayWriteArrayData>(file_path);
        return std::move(bind_data);
    } else {
        auto file_path = input.info.file_path;

        vector<uint64_t> tile_coords;
        if (dim_len == 1) {
            tile_coords.push_back(x);
        } else if (dim_len == 2) {
            tile_coords.push_back(x);
            tile_coords.push_back(y);
        } else if (dim_len == 3) {
            tile_coords.push_back(x);
            tile_coords.push_back(y);
            tile_coords.push_back(z);
        }

        auto bind_data =
            make_uniq<DenseToTileWriteArrayData>(file_path, tile_coords);
        return std::move(bind_data);
    }
}

static unique_ptr<LocalFunctionData> WriteArrayInitializeLocal(
    ExecutionContext &context, FunctionData &bind_data) {
    return std::move(make_uniq<LocalWriteArrayData>());
}

static unique_ptr<GlobalFunctionData> WriteArrayInitializeGlobal(
    ClientContext &context, FunctionData &bind_data, const string &file_path) {
    // std::cerr << "WriteArrayInitializeGlobal() called" << std::endl;
    return std::move(
        make_uniq<GlobalWriteArrayData>(context, bind_data, file_path));
}

void COOToArrayCopyArrayWriter::WriteArrayData(ExecutionContext &context,
                                               FunctionData &bind_data,
                                               GlobalFunctionData &gstate,
                                               LocalFunctionData &lstate,
                                               DataChunk &input) {
    // NOTE: I assume that only one thread runs
    auto &array_gstate = gstate.Cast<GlobalWriteArrayData>();
    auto &array_data = bind_data.Cast<COOToArrayWriteArrayData>();

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

static void WriteArraySink(ExecutionContext &context, FunctionData &bind_data,
                           GlobalFunctionData &gstate,
                           LocalFunctionData &lstate, DataChunk &input) {
    auto &array_gstate = gstate.Cast<GlobalWriteArrayData>();
    array_gstate.writer.get()->WriteArrayData(context, bind_data, gstate,
                                              lstate, input);
}

static void WriteArrayCombine(ExecutionContext &context,
                              FunctionData &bind_data,
                              GlobalFunctionData &gstate,
                              LocalFunctionData &lstate) {}

void WriteArrayFinalize(ClientContext &context, FunctionData &bind_data,
                        GlobalFunctionData &gstate) {
    auto &array_gstate = gstate.Cast<GlobalWriteArrayData>();
    array_gstate.unpin();
}

CopyFunctionExecutionMode WriteArrayExecutionMode(bool preserve_insertion_order,
                                                  bool supports_batch_index) {
    return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
}

struct WriteArrayBatchData : public PreparedBatchData {};

unique_ptr<PreparedBatchData> WriteArrayPrepareBatch(
    ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
    unique_ptr<ColumnDataCollection> collection) {
    return std::move(make_uniq<WriteArrayBatchData>());
}

//===--------------------------------------------------------------------===//
// Flush Batch
//===--------------------------------------------------------------------===//
void WriteArrayFlushBatch(ClientContext &context, FunctionData &bind_data,
                          GlobalFunctionData &gstate,
                          PreparedBatchData &batch) {}

idx_t WriteArrayFileSize(GlobalFunctionData &gstate) { return 1; }

CopyFunction ArrayExtension::GetCopyFunction() {
    std::cerr << "GetCopyFunction() called" << std::endl;

    CopyFunction info("tilestore");
    info.copy_to_bind = WriteArrayBind;
    info.copy_to_initialize_local = WriteArrayInitializeLocal;
    info.copy_to_initialize_global = WriteArrayInitializeGlobal;
    info.copy_to_sink = WriteArraySink;
    info.copy_to_combine = WriteArrayCombine;
    info.copy_to_finalize = WriteArrayFinalize;
    info.execution_mode = WriteArrayExecutionMode;
    // info.prepare_batch = WriteArrayPrepareBatch;
    // info.flush_batch = WriteArrayFlushBatch;
    info.file_size_bytes = WriteArrayFileSize;

    info.plan = nullptr;

    info.copy_from_bind = nullptr;
    info.copy_from_function = ArrayExtension::GetReadArrayFunction();

    info.extension = "tilestore";

    return info;
}

}  // namespace duckdb