
#include <chrono>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <thread>

#include "array_extension.hpp"
#include "array_reader.hpp"
#include "coo_reader.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "read_array.hpp"

extern "C" {
#include "bf.h"
}

namespace duckdb {

void CreateArrayScalarFunction(DataChunk &args, ExpressionState &state,
                               Vector &result) {
    // Getting vectors
    auto vec_arrname = FlatVector::GetData<string_t>(args.data[0]);
    // auto vec_arrsize = ListVector::GetData(args.data[1]);
    // auto vec_arrsize = ListVector::GetData(args.data[1]);
    // auto vec_tilesize = ListVector::GetData(args.data[2]);
    auto vec_type = FlatVector::GetData<string_t>(args.data[3]);
    auto vec_is_sparse = FlatVector::GetData<bool>(args.data[4]);

    // check the vector length is one
    if (args.size() != 1) {
        throw InternalException("The input size for create_array should be 1");
    }

    // data for lists
    auto d_arrsize = ListVector::GetEntry(args.data[1]);
    auto d_tilesize = ListVector::GetEntry(args.data[2]);

    // prepare for tilestore
    uint32_t dim_len = ListVector::GetListSize(args.data[1]);
    uint64_t *arr = new uint64_t[dim_len];
    uint64_t *tile = new uint64_t[dim_len];
    for (uint32_t i = 0; i < dim_len; i++) {
        arr[i] = d_arrsize.GetValue(i).GetValue<uint64_t>();
        tile[i] = d_tilesize.GetValue(i).GetValue<uint64_t>();
    }

    tilestore_datatype_t attr_type;
    if (strcmp(vec_type[0].GetData(), "FLOAT64") == 0) {
        attr_type = TILESTORE_FLOAT64;
    } else if (strcmp(vec_type[0].GetData(), "FLOAT32") == 0) {
        attr_type = TILESTORE_FLOAT32;
    } else if (strcmp(vec_type[0].GetData(), "UINT64") == 0) {
        attr_type = TILESTORE_UINT64;
    } else if (strcmp(vec_type[0].GetData(), "CHAR") == 0) {
        attr_type = TILESTORE_CHAR;
    } else {
        throw NotImplementedException("Unsupported type for create_array");
    }

    tilestore_format_t format = TILESTORE_DENSE;
    if (vec_is_sparse[0]) {
        format = TILESTORE_SPARSE_CSR;
    }

    string arrname;
    arrname.append(vec_arrname[0].GetData(), vec_arrname[0].GetSize());
    arrname.append("\0");

    // create array
    if (tilestore_create_array(arrname.c_str(), arr, tile, dim_len, attr_type,
                               format) != TILESTORE_OK) {
        throw InternalException("Failed to create array");
    }

    // FlatVector::GetData<bool>(result)[0] = true;

    UnaryExecutor::Execute<string_t, bool>(args.data[0], result, args.size(),
                                           [&](string_t name) { return true; });
}

ScalarFunction ArrayExtension::GetScalarFunction() {
    // this function should be called only once like "SELECT create_array(...)"
    ScalarFunction function = ScalarFunction(
        "create_array",
        {
            LogicalType::VARCHAR,                     // Array name
            LogicalType::LIST(LogicalType::INTEGER),  // Array size
            LogicalType::LIST(LogicalType::INTEGER),  // Tile size
            LogicalType::VARCHAR,                     // type
            LogicalType::BOOLEAN                      // is sparse array?
        },
        LogicalType::BOOLEAN, CreateArrayScalarFunction);

    return function;
}

}  // namespace duckdb
