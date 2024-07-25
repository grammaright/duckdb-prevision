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
#include "read_array.hpp"

extern "C" {
#include "bf.h"
}

namespace duckdb {
class ArrayReader {
   public:
    static uint64_t PutData(optional_ptr<const FunctionData> bind_data,
                            ArrayReadGlobalState &gstate, double *pagevals,
                            uint64_t size, DataChunk &output);

   private:
    static uint64_t _Put2DData(optional_ptr<const FunctionData> bind_data,
                               ArrayReadGlobalState &gstate, double *pagevals,
                               uint64_t size, DataChunk &output);
    static uint64_t _Put2DDataNoPrune(
        optional_ptr<const FunctionData> bind_data,
        ArrayReadGlobalState &gstate, double *pagevals, uint64_t size,
        DataChunk &output);
    static uint64_t _Put2DDataNoPruneAndProjection(
        optional_ptr<const FunctionData> bind_data,
        ArrayReadGlobalState &gstate, double *pagevals, uint64_t size,
        DataChunk &output);
    static uint64_t _Put1DData(optional_ptr<const FunctionData> bind_data,
                               ArrayReadGlobalState &gstate, double *pagevals,
                               uint64_t size, DataChunk &output);
    static uint64_t _Put1DDataNoPrune(
        optional_ptr<const FunctionData> bind_data,
        ArrayReadGlobalState &gstate, double *pagevals, uint64_t size,
        DataChunk &output);
    static uint64_t _Put1DDataNoPruneAndProjection(
        optional_ptr<const FunctionData> bind_data,
        ArrayReadGlobalState &gstate, double *pagevals, uint64_t size,
        DataChunk &output);
};
}  // namespace duckdb