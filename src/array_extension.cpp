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

extern "C"
{
#include "bf.h"
}

namespace duckdb
{

	static void LoadInternal(DatabaseInstance &instance)
	{
	}

	void ArrayExtension::Load(DuckDB &db)
	{
		std::cerr << "ArrayExtension::BF_Init()" << std::endl;

		BF_Init();
		BF_Attach();

		std::cerr << "define funtions" << std::endl;

		// BF_Detach();
		// BF_Free();
		// LoadInternal(*db.instance);
		// auto instance = ;
		auto table_function = ArrayExtension::GetTableFunction();
		// auto copy_function = ArrayExtension::GetCopyFunction();
		auto res = ArrayExtension::GetCopyFunction();

		std::cerr << "Registering functions" << std::endl;

		ExtensionUtil::RegisterFunction(*db.instance, table_function);
		ExtensionUtil::RegisterFunction(*db.instance, res);

		std::cerr << "ArrayExtension::Load()" << std::endl;
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
