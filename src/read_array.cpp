
#include <chrono>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <thread>

#include "array_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"

extern "C" {
#include "bf.h"
}

namespace duckdb {

struct ArrayReadData : public TableFunctionData {
   public:
    ArrayReadData(string arrayname) {
        uint64_t **_dim_domains;
        uint64_t *_tile_size;
        uint64_t *_array_size_in_tile;
        storage_util_get_dim_domains(arrayname.c_str(), &_dim_domains,
                                     &dim_len);
        storage_util_get_tile_extents(arrayname.c_str(), &_tile_size, &dim_len);
        storage_util_get_dcoord_lens(_dim_domains, _tile_size, dim_len,
                                     &_array_size_in_tile);

        array_size_in_tile = vector<uint64_t>(_array_size_in_tile,
                                              _array_size_in_tile + dim_len);
        tile_size = vector<uint64_t>(_tile_size, _tile_size + dim_len);

        storage_util_free_dim_domains(&_dim_domains, dim_len);
        storage_util_free_tile_extents(&_tile_size, dim_len);
        storage_util_free_dcoord_lens(&_array_size_in_tile);

        this->arrayname = arrayname;

        num_cells = 1;
        for (uint32_t i = 0; i < dim_len; i++) {
            num_cells *= tile_size[i];
        }
    };

   public:
    string arrayname;
    vector<uint64_t> array_size_in_tile;
    vector<uint64_t> tile_size;
    uint64_t num_cells;
    uint32_t dim_len;
};

struct ArrayReadGlobalState : public GlobalTableFunctionState {
   public:
    ArrayReadGlobalState(ClientContext &context,
                         TableFunctionInitInput &input) {
        auto &data = input.bind_data->Cast<ArrayReadData>();

        current_coords_in_tile = vector<uint64_t>(data.dim_len, 0);
        finished = false;

        // copy
        column_ids.assign(input.column_ids.begin(), input.column_ids.end());
        projection_ids.assign(input.projection_ids.begin(),
                              input.projection_ids.end());

        // get buffer
        auto dcoords = make_uniq_array<uint64_t>(2);
        for (uint32_t idx = 0; idx < data.dim_len; idx++) {
            dcoords[idx] = current_coords_in_tile[idx];
        }
        auto arrname = data.arrayname.c_str();
        _arrname_char = new char[1024];  // for char*
        strcpy(_arrname_char, arrname);

        // TODO: Consider sparse tile in the future
        array_key key = {_arrname_char, dcoords.get(), 2, BF_EMPTYTILE_DENSE};

        if (BF_GetBuf(key, &page) != BFE_OK) {
            throw InternalException("Failed to get buffer");
        }

        ArrayExtension::ResetPVBufferStats();
    };

    ~ArrayReadGlobalState() {
        auto dcoords = make_uniq_array<uint64_t>(2);
        for (uint32_t idx = 0; idx < 2; idx++) {
            dcoords[idx] = current_coords_in_tile[idx];
        }

        array_key key = {_arrname_char, dcoords.get(), 2, BF_EMPTYTILE_DENSE};
        BF_UnpinBuf(key);
        delete _arrname_char;

        ArrayExtension::PrintPVBufferStats();
    };

   public:
    uint64_t cell_idx;
    vector<uint64_t> current_coords_in_tile;
    bool finished;
    PFpage *page;

    // table function related:
    vector<column_t> column_ids;
    vector<idx_t> projection_ids;

   private:
    char *_arrname_char;
};

struct ArrayReadLocalState : public LocalTableFunctionState {
   public:
    ArrayReadLocalState(ClientContext &context, TableFunctionInitInput &input,
                        ArrayReadGlobalState &gstate){};
};

uint64_t Put2DData(optional_ptr<const FunctionData> bind_data,
                   ArrayReadGlobalState &gstate, double *pagevals,
                   uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE, total_remains);

    // for dest == 1 and 2
    for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
        auto dest = gstate.column_ids[gstate.projection_ids[i]];
        if (dest == 0) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t buf_idx = gstate.cell_idx + idx;
                uint32_t coord = buf_idx / data.tile_size[1];
                vec[idx] = coord;
            }
        } else if (dest == 1) {
            auto vec = FlatVector::GetData<uint32_t>(output.data[i]);
            for (uint64_t idx = 0; idx < local_remains; idx++) {
                uint32_t buf_idx = gstate.cell_idx + idx;
                uint32_t coord = buf_idx % data.tile_size[1];
                vec[idx] = coord;
            }
        } else if (dest == 2) {
            auto vec = FlatVector::GetData<double>(output.data[i]);
            memcpy(vec, pagevals + gstate.cell_idx,
                   sizeof(double) * local_remains);
        }
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t Put2DDataNoPrune(optional_ptr<const FunctionData> bind_data,
                          ArrayReadGlobalState &gstate, double *pagevals,
                          uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE, total_remains);

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t *coords;
        uint64_t buf_idx = gstate.cell_idx + idx;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), 2, &coords);

        double val = pagevals[buf_idx];

        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            if (gstate.column_ids[i] == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[0];
            else if (gstate.column_ids[i] == 1)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[1];
            else if (gstate.column_ids[i] == 2)
                FlatVector::GetData<double>(output.data[i])[idx] = val;
        }

        // std::cout << "\t[Put2DDataNoPrune] idx: " << idx << ", x: " <<
        // coords[0] << ", y: " << coords[1] << ", val: " << val << std::endl;
        free(coords);
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t Put2DDataNoPruneAndProjection(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE, total_remains);

    auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
    auto ys = FlatVector::GetData<uint32_t>(output.data[1]);
    auto vals =
        FlatVector::GetData<double>(output.data[2]);  // double type assumed

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t *coords;
        uint64_t buf_idx = gstate.cell_idx + idx;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), 2, &coords);

        double val = pagevals[buf_idx];

        xs[idx] = coords[0];
        ys[idx] = coords[1];
        vals[idx] = val;

        // std::cout << "\t[Put2DDataNoPruneAndProjection] idx: " << idx << ",
        // x: " << xs[idx] << ", y: " << ys[idx] << ", val: " << vals[idx] <<
        // std::endl;
        free(coords);
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t Put1DData(optional_ptr<const FunctionData> bind_data,
                   ArrayReadGlobalState &gstate, double *pagevals,
                   uint64_t size, DataChunk &output) {
    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE, total_remains);

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        auto buf_idx = gstate.cell_idx + idx;
        double val = pagevals[buf_idx];

        for (uint32_t i = 0; i < gstate.projection_ids.size(); i++) {
            auto dest = gstate.column_ids[gstate.projection_ids[i]];
            if (dest == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] = buf_idx;
            else if (dest == 1)
                FlatVector::GetData<double>(output.data[i])[idx] = val;
        }

        // std::cout << "\t[Put1DData] idx: " << idx << ", val: " << val <<
        // std::endl;
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}
uint64_t Put1DDataNoPrune(optional_ptr<const FunctionData> bind_data,
                          ArrayReadGlobalState &gstate, double *pagevals,
                          uint64_t size, DataChunk &output) {
    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE, total_remains);

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        auto buf_idx = gstate.cell_idx + idx;
        double val = pagevals[buf_idx];

        for (uint32_t i = 0; i < gstate.column_ids.size(); i++) {
            if (gstate.column_ids[i] == 0)
                FlatVector::GetData<uint32_t>(output.data[i])[idx] = buf_idx;
            else if (gstate.column_ids[i] == 1)
                FlatVector::GetData<double>(output.data[i])[idx] = val;
        }

        // std::cout << "\t[Put1DDataNoPrune] idx: " << idx << ", val: " << val
        // << std::endl;
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}
uint64_t Put1DDataNoPruneAndProjection(
    optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate,
    double *pagevals, uint64_t size, DataChunk &output) {
    auto total_remains = size - gstate.cell_idx;
    auto local_remains =
        std::min((uint64_t)STANDARD_VECTOR_SIZE, total_remains);

    auto &data = bind_data->Cast<ArrayReadData>();
    auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
    auto vals =
        FlatVector::GetData<double>(output.data[1]);  // double type assumed

    for (uint64_t idx = 0; idx < local_remains; idx++) {
        uint64_t *coords;
        uint64_t buf_idx = gstate.cell_idx + idx;
        bf_util_calculate_nd_from_1d_row_major(
            buf_idx, (uint64_t *)data.tile_size.data(), 2, &coords);

        double val = pagevals[buf_idx];
        xs[idx] = buf_idx;
        vals[idx] = val;

        // std::cout << "\t[Put1DDataNoPruneAndProjection] idx: " << idx << ",
        // val: " << vals[idx] << std::endl;

        free(coords);
    }

    gstate.cell_idx += local_remains;
    return local_remains;
}

uint64_t PutData(optional_ptr<const FunctionData> bind_data,
                 ArrayReadGlobalState &gstate, double *pagevals, uint64_t size,
                 DataChunk &output) {
    auto &data = bind_data->Cast<ArrayReadData>();

    if (data.dim_len == 2) {
        if (gstate.projection_ids.size() > 0)  // filter_prune ON
            return Put2DData(bind_data, gstate, pagevals, size, output);
        else if (gstate.column_ids.size() ==
                 output.data.size())  // no filter prune
            return Put2DDataNoPrune(bind_data, gstate, pagevals, size, output);
        else  // projection_pushdown and filter_prune are both false
            return Put2DDataNoPruneAndProjection(bind_data, gstate, pagevals,
                                                 size, output);
    } else {
        if (gstate.projection_ids.size() > 0)  // filter_prune ON
            return Put1DData(bind_data, gstate, pagevals, size, output);
        else if (gstate.column_ids.size() ==
                 output.data.size())  // no filter prune
            return Put1DDataNoPrune(bind_data, gstate, pagevals, size, output);
        else  // projection_pushdown and filter_prune are both false
            return Put1DDataNoPruneAndProjection(bind_data, gstate, pagevals,
                                                 size, output);
    }
}

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

    uint64_t read = PutData(data_p.bind_data, gstate, pagevals, size, output);

    output.SetCardinality(read);
    output.Verify();

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
    std::cout << "[ARRAY_EXTENSION] ReadArrayProgress: " << gstate.cell_idx
              << " / " << data.num_cells << " = " << progress << std::endl;
    return progress;
}

TableFunction ArrayExtension::GetTableFunction() {
    TableFunction function = TableFunction(
        "read_array",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::INTEGER)},
        ReadArrayFunction, ReadArrayBind, ReadArrayGlobalStateInit,
        ReadArrayLocalStateInit);
    // function.filter_pushdown = true;
    function.projection_pushdown = true;
    function.filter_prune = true;
    function.table_scan_progress = ReadArrayProgress;

    // function.pushdown_complex_filter = ReadArrayComplexFilterPushdown;
    // TODO: table_function.function_info = std::move(function_info);

    return function;
}
}  // namespace duckdb