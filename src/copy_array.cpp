#include "array_extension.hpp"
#include "copy_array.hpp"
#include "array_writer.hpp"

#include "buffer/bf.h"

namespace duckdb {

GlobalWriteArrayData::GlobalWriteArrayData(ClientContext &context,
                                           FunctionData &bind_data,
                                           const string &file_path) {
    auto &data = bind_data.Cast<CopyArrayData>();

    auto arrname = data.array_name.c_str();
    arrname_char = new char[1024];
    strcpy(arrname_char, arrname);

    dim_len = data.dim_len;
    tile_coords = new uint64_t[dim_len];

    is_pinned = false;
    page = NULL;

    // disable preemptive eviction for future array access (e.g., another table
    // or array queries)
    BF_DisablePE(arrname_char);

    ArrayExtension::ResetPVBufferStats();
}

void GlobalWriteArrayData::pin(vector<uint64_t> _tile_coords) {
    assert(!is_pinned);

    // getbuffer
    for (uint32_t i = 0; i < this->dim_len; i++) {
        tile_coords[i] = _tile_coords[i];
    }

    // determine dense or sparse
    emptytile_template_type_t emptytileTemplate = BF_EMPTYTILE_NONE;
    tilestore_format_t format;
    storage_util_get_array_type(arrname_char, &format);
    if (format == TILESTORE_DENSE) {
        emptytileTemplate = BF_EMPTYTILE_DENSE;
    } else if (format == TILESTORE_SPARSE_CSR) {
        emptytileTemplate = BF_EMPTYTILE_SPARSE_CSR;
    } else if (format == TILESTORE_SPARSE_COO) {
        emptytileTemplate = BF_EMPTYTILE_SPARSE_COO;
    }

    key = {arrname_char, tile_coords, dim_len, emptytileTemplate};

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
            } else if (mode_code == 2) {
                mode = COOMA_TO_COO;
            } else if (mode_code == 3) {
                mode = COOMA_TO_DENSE;
            } else {
                throw NotImplementedException("Unknown mode for WriteArray: %d",
                                              mode_code);
            }
        } else {
            throw NotImplementedException("Unknown option for WriteArray: %s",
                                          option.first.c_str());
        }
    }

    auto file_path = input.info.file_path;
    if (mode != VALUES_TO_TILE) {
        auto bind_data = make_uniq<CopyArrayData>(file_path, mode);
        return std::move(bind_data);
    } else {
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
            make_uniq<DenseToTileCopyArrayData>(file_path, mode, tile_coords);
        return std::move(bind_data);
    }
}

static unique_ptr<LocalFunctionData> WriteArrayInitializeLocal(
    ExecutionContext &context, FunctionData &bind_data) {
    return std::move(make_uniq<LocalWriteArrayData>());
}

static unique_ptr<GlobalFunctionData> WriteArrayInitializeGlobal(
    ClientContext &context, FunctionData &bind_data, const string &file_path) {
    auto &array_bind = bind_data.Cast<CopyArrayData>();
    if (array_bind.mode == VALUES_TO_TILE) {
        return std::move(make_uniq<GlobalDenseToTileWriteArrayData>(
            context, bind_data, file_path));
    } else if (array_bind.mode == COO_TO_ARRAY) {
        return std::move(make_uniq<GlobalCOOToArrayWriteArrayData>(
            context, bind_data, file_path));
    } else if (array_bind.mode == COOMA_TO_COO) {
        return std::move(make_uniq<GlobalCoomaToCooWriteArrayData>(
            context, bind_data, file_path));
    } else if (array_bind.mode == COOMA_TO_DENSE) {
        return std::move(make_uniq<GlobalCoomaToDenseWriteArrayData>(
            context, bind_data, file_path));
    }

    return nullptr;
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