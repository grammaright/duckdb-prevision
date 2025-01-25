
#include "read_array.hpp"

#include <chrono>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <thread>

#include "array_extension.hpp"
#include "array_reader.hpp"
#include "coo_reader.hpp"
#include "csr_reader.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"

#include "buffer/bf.h"

namespace duckdb {

static void ReadArrayFunctionWithCoords(ClientContext &context,
                                        TableFunctionInput &data_p, DataChunk &output) {
    auto &data = data_p.bind_data->Cast<ArrayReadData>();
    auto &gstate = data_p.global_state->Cast<ArrayReadGlobalState>();

    if (gstate.isFinished) {
        gstate.unpin();
        output.SetCardinality(0);
        output.Verify();
        return;
    }

    // read array
    auto size = gstate.page->unfilled_idx;
    char *pagevals = (char *)bf_util_get_pagebuf(gstate.page);
    uint64_t read = 0;

    // coords
    vector<uint64_t *> coords;
    for (int d = 0; d < (int)data.dim_len; d++) {
        coords.push_back(
            (uint64_t *)bf_util_pagebuf_get_coords(gstate.page, d));
    }

    if (data.format == TILESTORE_SPARSE_COO) {
        // TODO: Multi-tile array is not tested
        read = CooReader::PutData(data_p.bind_data, gstate, pagevals, coords,
                                  size, output);
    } else if (data.format == TILESTORE_SPARSE_CSR) {
        read = CsrReader::PutData(data_p.bind_data, gstate, pagevals, coords,
                                  size, output);
    } else {
        read = ArrayReader::PutData(data_p.bind_data, gstate, pagevals,
                                            size, output);
    }

    // set cardinality
    output.SetCardinality(read);
    output.Verify();

    // if we have read all cells, mark as finished
    if (gstate.cell_idx == size) {
        gstate.isFinished = true;
    }
}

static void ReadArrayFunctionAll(ClientContext &context,
                                        TableFunctionInput &data_p, DataChunk &output) {
    auto &data = data_p.bind_data->Cast<ArrayReadData>();
    auto &gstate = data_p.global_state->Cast<ArrayReadGlobalState>();

    if (gstate.isFinished) {
        gstate.unpin();
        output.SetCardinality(0);
        output.Verify();
        return;
    }

    while (!gstate.isFinished) {
        // read array
        auto size = gstate.page->unfilled_idx;
        char *pagevals = (char *)bf_util_get_pagebuf(gstate.page);
        uint64_t read = 0;

        if (data.format == TILESTORE_DENSE) {
            read = ArrayReader::PutData(data_p.bind_data, gstate, pagevals,
                                        size, output);
        } else {
            vector<uint64_t *> coords;
            for (int d = 0; d < (int)data.dim_len; d++) {
                coords.push_back(
                    (uint64_t *)bf_util_pagebuf_get_coords(gstate.page, d));
            }

            if (data.format == TILESTORE_SPARSE_COO) {
                read = CooReader::PutData(data_p.bind_data, gstate, pagevals,
                                          coords, size, output);
            } else {
                read = CsrReader::PutData(data_p.bind_data, gstate, pagevals,
                                          coords, size, output);
            }
        }

        // set cardinality
        output.SetCardinality(output.size() + read);
        output.Verify();

        // if not finished but vector is full, escape the loop and read again in
        // the next function call
        if (gstate.cell_idx < size) {
            break;
        }

        // if finished, move to the next tile. If there is no tile, isFinish
        // will be set to true
        gstate.findNextTile();
    }

    // std::cerr << "done function call" << std::endl;
}

static void ReadArrayFunction(ClientContext &context,
                              TableFunctionInput &data_p, DataChunk &output) {

    auto &data = data_p.bind_data->Cast<ArrayReadData>();
    // auto &gstate = data_p.global_state->Cast<ArrayReadGlobalState>();

    if (data.requestedCoords.size() == 0) {
        ReadArrayFunctionAll(context, data_p, output);
    } else {
        ReadArrayFunctionWithCoords(context, data_p, output);
    }
}

// static LogicalType GetStringToType(string attr_type) {
//     if (attr_type == "INTEGER") {
//         return LogicalType::INTEGER;
//     } else if (attr_type == "FLOAT") {
//         return LogicalType::FLOAT;
//     } else if (attr_type == "DOUBLE") {
//         return LogicalType::DOUBLE;
//     } else {
//         throw NotImplementedException("Unsupported attribute type: " + attr_type);
//     }
//     return LogicalType::INVALID;
// }

unique_ptr<FunctionData> ReadArrayBind(ClientContext &context,
                                       TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
    // TODO: process input tile coordinates
    string arrayname = StringValue::Get(input.inputs[0]);
    unique_ptr<ArrayReadData> bind_data = make_uniq<ArrayReadData>(arrayname);

    // array_type
    for (auto it = input.named_parameters.begin();
         it != input.named_parameters.end(); it++) {
        std::cerr << "Named parameter: " << it->first << " = " << it->second
                  << std::endl;
        if (it->first == "coords") {
            auto coords = ListValue::GetChildren(it->second);
            if (coords.size() != bind_data->dim_len) {
                throw std::runtime_error("Invalid number of coordinates");
            }

            for (int d = 0; d < (int) bind_data->dim_len; d++) {
                bind_data->requestedCoords.push_back(
                    coords[d].GetValue<int64_t>());
            }
        }
    }

    // TODO: multi-attributes are not supported yet for dense array
    bool isMultiAttrs = bind_data->attrTypes.size() > 1;
    if (bind_data->format == TILESTORE_DENSE && isMultiAttrs) {
        throw NotImplementedException("Multi-attributes are not supported yet");
    }

    // dimensions first
    if (bind_data->dim_len == 1) {
        return_types.push_back(LogicalType::UINTEGER);
        names.emplace_back("x");
    } else if (bind_data->dim_len == 2) {
        return_types.push_back(LogicalType::UINTEGER);
        return_types.push_back(LogicalType::UINTEGER);

        names.emplace_back("x");
        names.emplace_back("y");
    } else if (bind_data->dim_len == 3) {
        return_types.push_back(LogicalType::UINTEGER);
        return_types.push_back(LogicalType::UINTEGER);
        return_types.push_back(LogicalType::UINTEGER);

        names.emplace_back("x");
        names.emplace_back("y");
        names.emplace_back("z");
    } else {
        throw NotImplementedException(
            "Only 1D, 2D, and 3D arrays are supported");
    }

    // attributes
    for (uint32_t i = 0; i < bind_data->attrTypes.size(); i++) {
        return_types.push_back(bind_data->attrTypes[i]);
        names.emplace_back("a" + std::to_string(i));
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

unique_ptr<NodeStatistics> ReadArrayCardinality(ClientContext &context,
                                                const FunctionData *bind_data) {
    auto stat = make_uniq<NodeStatistics>();
    stat->estimated_cardinality = 1;

    auto &array_data = bind_data->Cast<ArrayReadData>();
    for (uint32_t idx = 0; idx < array_data.dim_len; idx++) {
        stat->estimated_cardinality *= array_data.tile_size[idx];
    }

    stat->max_cardinality = stat->estimated_cardinality;

    stat->has_estimated_cardinality = true;
    stat->has_max_cardinality = true;
    return std::move(stat);
}

TableFunction ArrayExtension::GetReadArrayFunction() {
    TableFunction function = TableFunction(
        "read_array",
        {LogicalType::VARCHAR},
        ReadArrayFunction, ReadArrayBind, ReadArrayGlobalStateInit,
        ReadArrayLocalStateInit);

    function.named_parameters["coords"] =
        LogicalType::LIST(LogicalType::INTEGER);
    function.named_parameters["array_type"] = LogicalType::VARCHAR;
    function.named_parameters["attr_types"] =
        LogicalType::LIST(LogicalType::VARCHAR);

    // function.filter_pushdown = true;
    function.projection_pushdown = true;
    function.filter_prune = true;
    function.table_scan_progress = ReadArrayProgress;
    function.cardinality = ReadArrayCardinality;

    // function.pushdown_complex_filter = ReadArrayComplexFilterPushdown;
    // TODO: table_function.function_info = std::move(function_info);

    return function;
}

}  // namespace duckdb