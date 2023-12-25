#define DUCKDB_EXTENSION_MAIN

#include "quack_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

extern "C" {
#include "bf.h"
}

namespace duckdb {

struct ArrayScanData : public TableFunctionData {
	string arrayname;
};

struct ArrayScanGlobalState {
	// ArrayScanData data;
};

struct ArrayScanLocalState {};

struct ArrayGlobalTableFunctionState : public GlobalTableFunctionState {
public:
	ArrayGlobalTableFunctionState(ClientContext &context, TableFunctionInitInput &input) {};
	static unique_ptr<GlobalTableFunctionState> Init(ClientContext &context, TableFunctionInitInput &input) {
		BF_Init();
		BF_Attach();

		return std::move(make_uniq<ArrayGlobalTableFunctionState>(context, input));
	}

public:
	ArrayScanGlobalState state;
};

struct ArrayLocalTableFunctionState : public LocalTableFunctionState {
public:
	ArrayLocalTableFunctionState(ClientContext &context, ArrayScanGlobalState &gstate) {};
	static unique_ptr<LocalTableFunctionState> Init(ExecutionContext &context, TableFunctionInitInput &input,
	                                                GlobalTableFunctionState *global_state) {
		return std::move(make_uniq<ArrayLocalTableFunctionState>(context.client, global_state->Cast<ArrayGlobalTableFunctionState>().state));
	};

public:
	ArrayScanLocalState state;
};

inline void QuackScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "Quack "+name.GetString()+" 🐥");;
        });
}

static void ReadArrayFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &gstate = data_p.global_state->Cast<ArrayGlobalTableFunctionState>().state;
	auto &lstate = data_p.local_state->Cast<ArrayLocalTableFunctionState>().state;

	// FIXME:
	static bool first = true;
	if (!first) {
		output.SetCardinality(0);
		output.Verify();
		return;
	}
	first = false;

	auto dcoords = make_uniq_array<uint64_t>(2);
	dcoords[0] = 0, dcoords[1] = 0;
	auto arrname = ((ArrayScanData*) data_p.bind_data.get())->arrayname.c_str();
	char *arrname_char = new char[1024];
	strcpy(arrname_char, arrname);
	
	// FIXME: need to get coordinates
	array_key key = {arrname_char, "a", dcoords.get(), 2, BF_EMPTYTILE_DENSE};
	PFpage *page;
	BF_GetBuf(key, &page);

	// FIXME: type!!
	auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
	auto ys = FlatVector::GetData<uint32_t>(output.data[1]);
	auto vals = FlatVector::GetData<double>(output.data[2]);

	auto size = page->pagebuf_len / sizeof(double);
	double *pageval = (double*) bf_util_get_pagebuf(page);
	for (uint64_t idx = 0; idx < size; idx++) {
		uint64_t *coords;
		bf_util_calculate_nd_from_1d_row_major(idx, bf_util_get_tile_extents(page), 2, &coords);

		double val = pageval[idx];

		xs[idx] = coords[0];
		ys[idx] = coords[1];
		vals[idx] = val;

		free(coords);
	}
	
	output.SetCardinality(size);
	output.Verify();

	BF_UnpinBuf(key);
	delete arrname_char;

	// if (output.size() != 0) {
		// MultiFileReader::FinalizeChunk(gstate.bind_data.reader_bind, lstate.GetReaderData(), output);
	// }
}


unique_ptr<FunctionData> ReadArrayBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<ArrayScanData>();
	bind_data->arrayname = StringValue::Get(input.inputs[0]);

	return_types.push_back(LogicalType::UINTEGER);
	return_types.push_back(LogicalType::UINTEGER);
	return_types.push_back(LogicalType::DOUBLE);
	
	names.emplace_back("x");
	names.emplace_back("y");
	names.emplace_back("val");

	return std::move(bind_data);
}

static void LoadInternal(DatabaseInstance &instance) {
    // Register a scalar function
    auto quack_scalar_function = ScalarFunction("quack", {LogicalType::VARCHAR}, LogicalType::VARCHAR, QuackScalarFun);
    ExtensionUtil::RegisterFunction(instance, quack_scalar_function);

    TableFunction function = TableFunction(
        "read_array", {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::INTEGER)}, 
        ReadArrayFunction, ReadArrayBind, ArrayGlobalTableFunctionState::Init, ArrayLocalTableFunctionState::Init);
	// TODO: table_function.function_info = std::move(function_info);

    ExtensionUtil::RegisterFunction(instance, function);
}

void QuackExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string QuackExtension::Name() {
	return "quack";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void quack_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::QuackExtension>();
}

DUCKDB_EXTENSION_API const char *quack_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
