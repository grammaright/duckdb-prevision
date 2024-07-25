
#include "read_array.hpp"

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

extern "C" {
#include "bf.h"
}

namespace duckdb {

static void ReadArrayFunction(ClientContext &context,
                              TableFunctionInput &data_p, DataChunk &output) {
    auto &data = data_p.bind_data->Cast<ArrayReadData>();
    auto &gstate = data_p.global_state->Cast<ArrayReadGlobalState>();

    // check if we have already read all the tiles
    if (gstate.finished) {
        output.SetCardinality(0);
        output.Verify();
        return;
    }

    auto size = gstate.page->pagebuf_len / sizeof(double);
    double *pagevals = (double *)bf_util_get_pagebuf(gstate.page);

    if (data.is_coo_array) {
        // TODO: Multi-tile array is not tested
        uint64_t read = CooReader::PutData(data_p.bind_data, gstate, pagevals,
                                           size / (data.dim_len + 1), output);

        output.SetCardinality(read);
        output.Verify();
    } else {
        uint64_t read = ArrayReader::PutData(data_p.bind_data, gstate, pagevals,
                                             size, output);

        output.SetCardinality(read);
        output.Verify();
    }

    // move the current_coords_in_tile to the next tile
    if (gstate.cell_idx < size) return;

    for (int64_t idx = data.dim_len - 1; idx >= 0; idx--) {
        gstate.current_coords_in_tile[idx] += 1;
        if (gstate.current_coords_in_tile[idx] < data.array_size_in_tile[idx]) {
            break;
        }

        gstate.current_coords_in_tile[idx] = 0;
        gstate.cell_idx = 0;
        if (idx == 0) {
            gstate.finished = true;
            gstate.free();
        }
    }
}

unique_ptr<FunctionData> ReadArrayBind(ClientContext &context,
                                       TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
    // TODO: process input tile coordinates
    string arrayname = StringValue::Get(input.inputs[0]);
    unique_ptr<ArrayReadData> bind_data = make_uniq<ArrayReadData>(arrayname);

    // array_type
    bool is_coo_array = false;
    for (auto it = input.named_parameters.begin();
         it != input.named_parameters.end(); it++) {
        std::cerr << "Named parameter: " << it->first << " = " << it->second
                  << std::endl;
        if (it->first == "array_type" && it->second == "COO") {
            is_coo_array = true;
        }
    }

    bind_data->is_coo_array = is_coo_array;

    // TODO: Currently, only 2D array is supported for COO array
    if (is_coo_array) {
        assert(bind_data->dim_len == 2);
    }

    // regular array (dense & sparse (in the future))
    if (bind_data->dim_len == 1) {
        return_types.push_back(LogicalType::UINTEGER);
        return_types.push_back(LogicalType::DOUBLE);

        names.emplace_back("idx");
        names.emplace_back("val");
    } else if (bind_data->dim_len == 2) {
        return_types.push_back(LogicalType::UINTEGER);
        return_types.push_back(LogicalType::UINTEGER);
        return_types.push_back(LogicalType::DOUBLE);

        names.emplace_back("x");
        names.emplace_back("y");
        names.emplace_back("val");
    } else {
        throw NotImplementedException("Only 1D and 2D arrays are supported");
    }

    return std::move(bind_data);
}

unique_ptr<GlobalTableFunctionState> ReadArrayGlobalStateInit(
    ClientContext &context, TableFunctionInitInput &input) {
    return std::move(make_uniq<ArrayReadGlobalState>(context, input));
}

unique_ptr<LocalTableFunctionState> ReadArrayLocalStateInit(
    ExecutionContext &context, TableFunctionInitInput &input,
    GlobalTableFunctionState *gstate) {
    return std::move(make_uniq<ArrayReadLocalState>(
        context.client, input, gstate->Cast<ArrayReadGlobalState>()));
}

// void ReadArrayComplexFilterPushdown(
//     ClientContext &context, LogicalGet &get, FunctionData *bind_data_p,
//     vector<unique_ptr<Expression>> &filters)
// {
//     std::cerr << "[" << pthread_self()
//               << "]"
//                  "CSVComplexFilterPushdown"
//               << std::endl;
// }

double ReadArrayProgress(ClientContext &context, const FunctionData *bind_data,
                         const GlobalTableFunctionState *global_state) {
    auto gstate = global_state->Cast<ArrayReadGlobalState>();
    auto data = bind_data->Cast<ArrayReadData>();

    auto progress = (double)gstate.cell_idx / data.num_cells;
    // std::cout << "[ARRAY_EXTENSION] ReadArrayProgress: " << gstate.cell_idx
    //           << " / " << data.num_cells << " = " << progress << std::endl;
    return progress;
}

TableFunction ArrayExtension::GetTableFunction() {
    TableFunction function = TableFunction(
        "read_array",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::INTEGER)},
        ReadArrayFunction, ReadArrayBind, ReadArrayGlobalStateInit,
        ReadArrayLocalStateInit);

    function.named_parameters["array_type"] = LogicalType::VARCHAR;
    // function.filter_pushdown = true;
    function.projection_pushdown = true;
    function.filter_prune = true;
    function.table_scan_progress = ReadArrayProgress;

    // function.pushdown_complex_filter = ReadArrayComplexFilterPushdown;
    // TODO: table_function.function_info = std::move(function_info);

    return function;
}
}  // namespace duckdb