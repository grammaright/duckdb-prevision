#define DUCKDB_EXTENSION_MAIN

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include "duckdb/function/copy_function.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"

#include "array_extension.hpp"
#include <sys/stat.h>

extern "C"
{
#include "bf.h"
}

namespace duckdb
{

	bool IsBFInitialized()
	{
		struct stat buf;
		int ret = stat("/dev/shm/buffertile_bf", &buf);
		return (ret == 0);
	}

	void ArrayExtension::Load(DuckDB &db)
	{
		std::cerr << "ArrayExtension::Load()" << std::endl;

		if (!IsBFInitialized())
		{
			// It will run only if in the development of DuckDB extension
			BF_Init();
			fprintf(stderr, "[ARRAY_EXT] BF_Init() is called because bf has not been initialized.\n");
		}

		BF_Attach();

		std::cerr << "define funtions" << std::endl;
		auto table_function = ArrayExtension::GetTableFunction();
		auto copy_function = ArrayExtension::GetCopyFunction();

		std::cerr << "Registering functions" << std::endl;
		ExtensionUtil::RegisterFunction(*db.instance, table_function);
		ExtensionUtil::RegisterFunction(*db.instance, copy_function);
	}
	std::string ArrayExtension::Name()
	{
		std::cout << "ArrayExtension::Name()" << std::endl;
		return "array";
	}

} // namespace duckdb

extern "C"
{
	DUCKDB_EXTENSION_API void array_init(duckdb::DatabaseInstance &db)
	{
		std::cout << "array_init()" << std::endl;

		duckdb::DuckDB db_wrapper(db);
		db_wrapper.LoadExtension<duckdb::ArrayExtension>();
	}

	DUCKDB_EXTENSION_API const char *array_version()
	{
		std::cout << "array_version()" << std::endl;
		return duckdb::DuckDB::LibraryVersion();
	}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
