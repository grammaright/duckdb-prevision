#pragma once

#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <limits>

#include "array_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/bind_helpers.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/serializer/write_stream.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/function/scalar/string_functions.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"

#include "buffer/bf.h"

namespace duckdb {

typedef enum ArrayCopyFunctionExecutionMode {
    COO_TO_ARRAY = 0,
    VALUES_TO_TILE = 1,
    COOMA_TO_COO = 2,
    COOMA_TO_DENSE = 3
} ArrayCopyFunctionExecutionMode;

class CopyArrayWriter;

struct GlobalWriteArrayData : public GlobalFunctionData {
   public:
    GlobalWriteArrayData(ClientContext &context, FunctionData &bind_data,
                         const string &file_path);

    ~GlobalWriteArrayData() {
        // FIXME: why double free?
        // BF_UnpinBuf(key);
        // delete arrname_char;
        // delete dcoords;

        // ArrayExtension::PrintPVBufferStats();
    }

    void pin(vector<uint64_t> tile_coords);
    void unpin();

    // uint32_t *GetTileCoords();

   public:
    unique_ptr<CopyArrayWriter> writer;

    PFpage *page;
    uint64_t buf_size;
    double *buf;

    uint64_t *tile_coords;
    uint32_t dim_len;
    bool is_pinned;

   protected:
    array_key key;
    char *arrname_char;

    uint64_t rowIdx;        // only for CoomaToCooCopyArrayWriter

    // to check the tile has been pinned to set nullbits
    set<vector<uint64_t>> pinned_tiles;
};

struct LocalWriteArrayData : public LocalFunctionData {};

}  // namespace duckdb