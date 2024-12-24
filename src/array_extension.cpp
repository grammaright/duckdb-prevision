#define DUCKDB_EXTENSION_MAIN

#include "array_extension.hpp"

#include <sys/stat.h>

#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"

#include "buffer/bf.h"

extern unsigned long long bftime, bf_this_query;

extern unsigned long long bf_tmpbuf_cnt;
// extern unsigned long long malloc_time;
extern unsigned long long bf_read_io_time, bf_iread_io_time, bf_read_io_size,
    bf_iread_io_size;
extern unsigned long long bf_write_io_time, bf_write_io_size;
extern unsigned long long bf_getbuf_cnt_hit, bf_getbuf_cnt_total;
extern unsigned long long bf_getbuf_io_hit, bf_getbuf_io_total;
extern unsigned long long bf_min_sl_update_time, bf_min_fl_retrival_time;

namespace duckdb {

bool IsBFInitialized() {
    struct stat buf;
    int ret = stat("/dev/shm/buffertile_bf", &buf);
    return (ret == 0);
}

void ArrayExtension::Load(DuckDB &db) {
    std::cerr << "ArrayExtension::Load()" << std::endl;

    if (!IsBFInitialized()) {
        // It will run only if in the development of DuckDB extension
        BF_Init();
        fprintf(stderr,
                "[ARRAY_EXT] BF_Init() is called because bf has not been "
                "initialized.\n");
    }

    // When debuging: BF_Init() and Attached() will be called in
    //    sqlite3_api_wrapper.cpp
    // When using M2: this will be called
    if (_mspace_data == nullptr) {
        BF_Attach();
    }

    std::cerr << "define funtions" << std::endl;
    auto table_function = ArrayExtension::GetReadArrayFunction();
    auto copy_function = ArrayExtension::GetCopyFunction();
    auto scalar_function = ArrayExtension::GetCreateArrayFunction();
    auto bfree_function = ArrayExtension::GetBfFreeFunction();

    std::cerr << "Registering functions" << std::endl;
    ExtensionUtil::RegisterFunction(*db.instance, table_function);
    ExtensionUtil::RegisterFunction(*db.instance, copy_function);
    ExtensionUtil::RegisterFunction(*db.instance, scalar_function);
    ExtensionUtil::RegisterFunction(*db.instance, bfree_function);
}

void BFFreeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    BF_Detach();
    BF_Free();

    // Do nothing
    auto vec = FlatVector::GetData<bool>(result);
    vec[0] = true;
    result.SetVectorType(VectorType::CONSTANT_VECTOR);
}

ScalarFunction ArrayExtension::GetBfFreeFunction() {
    return ScalarFunction("bf_free", {}, LogicalType::BOOLEAN, BFFreeFunction);
}

std::string ArrayExtension::Name() {
    std::cout << "ArrayExtension::Name()" << std::endl;
    return "array";
}

void ArrayExtension::ResetPVBufferStats() {
    bftime = 0;
    bf_this_query = 0;
    bf_tmpbuf_cnt = 0;
    //   malloc_time = 0;
    bf_read_io_time = 0;
    bf_iread_io_time = 0;
    bf_read_io_size = 0;
    bf_iread_io_size = 0;
    bf_write_io_time = 0;
    bf_write_io_size = 0;
    bf_getbuf_cnt_hit = 0;
    bf_getbuf_cnt_total = 0;
    bf_getbuf_io_hit = 0;
    bf_getbuf_io_total = 0;
    bf_min_sl_update_time = 0;
    bf_min_fl_retrival_time = 0;
}

void ArrayExtension::PrintPVBufferStats() {
    std::cerr << "total\tbf\tio_r\tio_w"
                 "\tphit\tpreq\tflgen\tflget\tsl\tdelhint\tpureplan\n"
              << std::endl;
    std::cerr << bf_this_query << "\t" << bftime << "\t" << bf_read_io_time
              << "\t" << bf_write_io_time << "\t" << bf_getbuf_cnt_hit << "\t"
              << bf_getbuf_cnt_total << "\t" << bf_min_fl_retrival_time << "\t"
              << bf_min_sl_update_time << "\t" << 0 << "\t" << 0 << "\t" << 0
              << std::endl;
    std::cerr << "0\t0\t" << bf_read_io_size << "\t" << bf_write_io_size << "\t"
              << bf_getbuf_io_hit << "\t" << bf_getbuf_io_total << "\t0\t0\t0"
              << std::endl;
}

}  // namespace duckdb

extern "C" {
DUCKDB_EXTENSION_API void array_init(duckdb::DatabaseInstance &db) {
    std::cout << "array_init()" << std::endl;

    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::ArrayExtension>();
}

DUCKDB_EXTENSION_API const char *array_version() {
    std::cout << "array_version()" << std::endl;
    return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
