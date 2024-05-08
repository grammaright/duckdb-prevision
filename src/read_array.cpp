
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include "array_extension.hpp"

extern "C"
{
#include "bf.h"
}

namespace duckdb
{

    struct ArrayReadData : public TableFunctionData
    {
    public:
        ArrayReadData(string arrayname)
        {
            uint64_t **_dim_domains;
            uint64_t *_tile_size;
            uint64_t *_array_size_in_tile;
            storage_util_get_dim_domains(arrayname.c_str(), &_dim_domains, &dim_len);
            storage_util_get_tile_extents(arrayname.c_str(), &_tile_size, &dim_len);
            storage_util_get_dcoord_lens(_dim_domains, _tile_size, dim_len, &_array_size_in_tile);

            array_size_in_tile = vector<uint64_t>(_array_size_in_tile, _array_size_in_tile + dim_len);
            tile_size = vector<uint64_t>(_tile_size, _tile_size + dim_len);

            storage_util_free_dim_domains(&_dim_domains, dim_len);
            storage_util_free_tile_extents(&_tile_size, dim_len);
            storage_util_free_dcoord_lens(&_array_size_in_tile);

            this->arrayname = arrayname;
        };

    public:
        string arrayname;
        vector<uint64_t> array_size_in_tile;
        vector<uint64_t> tile_size;
        uint32_t dim_len;
    };

    struct ArrayReadGlobalState : public GlobalTableFunctionState
    {
    public:
        ArrayReadGlobalState(ClientContext &context, TableFunctionInitInput &input)
        {
            auto &data = input.bind_data->Cast<ArrayReadData>();

            current_coords_in_tile = vector<uint64_t>(data.dim_len, 0);
            finished = false;

            // copy
            column_ids.assign(input.column_ids.begin(), input.column_ids.end());
            projection_ids.assign(input.projection_ids.begin(), input.projection_ids.end());
        };

        ~ArrayReadGlobalState(){};

    public:
        vector<uint64_t> current_coords_in_tile;
        bool finished;

        // table function related:
        vector<column_t> column_ids;
        vector<idx_t> projection_ids;
    };

    struct ArrayReadLocalState : public LocalTableFunctionState
    {
    public:
        ArrayReadLocalState(ClientContext &context, TableFunctionInitInput &input, ArrayReadGlobalState &gstate){};
    };

    void Put2DData(optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate, double *pagevals, uint64_t size, DataChunk &output)
    {
        auto &data = bind_data->Cast<ArrayReadData>();
        for (uint64_t idx = 0; idx < size; idx++)
        {
            uint64_t *coords;
            bf_util_calculate_nd_from_1d_row_major(idx, (uint64_t *)data.tile_size.data(), 2, &coords);

            double val = pagevals[idx];

            for (uint32_t i = 0; i < gstate.projection_ids.size(); i++)
            {
                auto dest = gstate.column_ids[gstate.projection_ids[i]];
                if (dest == 0)
                    FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[0];
                else if (dest == 1)
                    FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[1];
                else if (dest == 2)
                    FlatVector::GetData<double>(output.data[i])[idx] = val;
            }

            std::cout << "\t[Put2DData] idx: " << idx << ", x: " << coords[0] << ", y: " << coords[1] << ", val: " << val << std::endl;
            free(coords);
        }
    }
    void Put2DDataNoPrune(optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate, double *pagevals, uint64_t size, DataChunk &output)
    {
        auto &data = bind_data->Cast<ArrayReadData>();
        for (uint64_t idx = 0; idx < size; idx++)
        {
            uint64_t *coords;
            bf_util_calculate_nd_from_1d_row_major(idx, (uint64_t *)data.tile_size.data(), 2, &coords);

            double val = pagevals[idx];

            for (uint32_t i = 0; i < gstate.column_ids.size(); i++)
            {
                if (gstate.column_ids[i] == 0)
                    FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[0];
                else if (gstate.column_ids[i] == 1)
                    FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[1];
                else if (gstate.column_ids[i] == 2)
                    FlatVector::GetData<double>(output.data[i])[idx] = val;
            }

            std::cout << "\t[Put2DDataNoPrune] idx: " << idx << ", x: " << coords[0] << ", y: " << coords[1] << ", val: " << val << std::endl;
            free(coords);
        }
    }
    void Put2DDataNoPruneAndProjection(optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate, double *pagevals, uint64_t size, DataChunk &output)
    {
        auto &data = bind_data->Cast<ArrayReadData>();
        auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
        auto ys = FlatVector::GetData<uint32_t>(output.data[1]);
        auto vals = FlatVector::GetData<double>(output.data[2]); // double type assumed

        for (uint64_t idx = 0; idx < size; idx++)
        {
            uint64_t *coords;
            bf_util_calculate_nd_from_1d_row_major(idx, (uint64_t *)data.tile_size.data(), 2, &coords);

            double val = pagevals[idx];

            xs[idx] = coords[0];
            ys[idx] = coords[1];
            vals[idx] = val;

            std::cout << "\t[Put2DDataNoPruneAndProjection] idx: " << idx << ", x: " << xs[idx] << ", y: " << ys[idx] << ", val: " << vals[idx] << std::endl;

            free(coords);
        }
    }

    void Put1DData(optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate, double *pagevals, uint64_t size, DataChunk &output)
    {
        for (uint64_t idx = 0; idx < size; idx++)
        {
            double val = pagevals[idx];

            for (uint32_t i = 0; i < gstate.projection_ids.size(); i++)
            {
                auto dest = gstate.column_ids[gstate.projection_ids[i]];
                if (dest == 0)
                    FlatVector::GetData<uint32_t>(output.data[i])[idx] = idx;
                else if (dest == 1)
                    FlatVector::GetData<double>(output.data[i])[idx] = val;
            }

            std::cout << "\t[Put1DData] idx: " << idx << ", val: " << val << std::endl;
        }
    }
    void Put1DDataNoPrune(optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate, double *pagevals, uint64_t size, DataChunk &output)
    {
        for (uint64_t idx = 0; idx < size; idx++)
        {
            double val = pagevals[idx];

            for (uint32_t i = 0; i < gstate.column_ids.size(); i++)
            {
                if (gstate.column_ids[i] == 0)
                    FlatVector::GetData<uint32_t>(output.data[i])[idx] = idx;
                else if (gstate.column_ids[i] == 1)
                    FlatVector::GetData<double>(output.data[i])[idx] = val;
            }

            std::cout << "\t[Put1DDataNoPrune] idx: " << idx << ", val: " << val << std::endl;
        }
    }
    void Put1DDataNoPruneAndProjection(optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate, double *pagevals, uint64_t size, DataChunk &output)
    {
        auto &data = bind_data->Cast<ArrayReadData>();
        auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
        auto vals = FlatVector::GetData<double>(output.data[1]); // double type assumed

        for (uint64_t idx = 0; idx < size; idx++)
        {
            uint64_t *coords;
            bf_util_calculate_nd_from_1d_row_major(idx, (uint64_t *)data.tile_size.data(), 2, &coords);

            double val = pagevals[idx];
            xs[idx] = idx;
            vals[idx] = val;

            std::cout << "\t[Put1DDataNoPruneAndProjection] idx: " << idx << ", val: " << vals[idx] << std::endl;

            free(coords);
        }
    }

    void PutData(optional_ptr<const FunctionData> bind_data, ArrayReadGlobalState &gstate, double *pagevals, uint64_t size, DataChunk &output)
    {
        auto &data = bind_data->Cast<ArrayReadData>();

        if (data.dim_len == 2)
        {
            if (gstate.projection_ids.size() > 0) // filter_prune ON
                Put2DData(bind_data, gstate, pagevals, size, output);
            else if (gstate.column_ids.size() == output.data.size()) // no filter prune
                Put2DDataNoPrune(bind_data, gstate, pagevals, size, output);
            else // projection_pushdown and filter_prune are both false
                Put2DDataNoPruneAndProjection(bind_data, gstate, pagevals, size, output);
        }
        else
        {
            if (gstate.projection_ids.size() > 0) // filter_prune ON
                Put1DData(bind_data, gstate, pagevals, size, output);
            else if (gstate.column_ids.size() == output.data.size()) // no filter prune
                Put1DDataNoPrune(bind_data, gstate, pagevals, size, output);
            else // projection_pushdown and filter_prune are both false
                Put1DDataNoPruneAndProjection(bind_data, gstate, pagevals, size, output);
        }
    }

    static void ReadArrayFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output)
    {
        auto &data = data_p.bind_data->Cast<ArrayReadData>();
        auto &gstate = data_p.global_state->Cast<ArrayReadGlobalState>();

        // check if we have already read all the tiles
        if (gstate.finished)
        {
            output.SetCardinality(0);
            output.Verify();
            return;
        }

        // otherwise, read the current tile
        auto dcoords = make_uniq_array<uint64_t>(2);
        for (uint32_t idx = 0; idx < data.dim_len; idx++)
        {
            dcoords[idx] = gstate.current_coords_in_tile[idx];
        }
        auto arrname = data.arrayname.c_str();
        // why allocate newly?
        char *arrname_char = new char[1024];
        strcpy(arrname_char, arrname);

        // TODO: Consider sparse tile in the future
        array_key key = {arrname_char, (char *)"a", dcoords.get(), 2, BF_EMPTYTILE_DENSE};

        PFpage *page;
        if (BF_GetBuf(key, &page) != BFE_OK)
        {
            throw InternalException("Failed to get buffer");
        }

        auto size = page->pagebuf_len / sizeof(double);
        double *pagevals = (double *)bf_util_get_pagebuf(page);

        PutData(data_p.bind_data, gstate, pagevals, size, output);

        output.SetCardinality(size);
        output.Verify();

        BF_UnpinBuf(key);
        delete arrname_char;

        // move the current_coords_in_tile to the next tile
        for (int64_t idx = data.dim_len - 1; idx >= 0; idx--)
        {
            gstate.current_coords_in_tile[idx] += 1;
            if (gstate.current_coords_in_tile[idx] < data.array_size_in_tile[idx])
            {
                break;
            }

            gstate.current_coords_in_tile[idx] = 0;
            if (idx == 0)
            {
                gstate.finished = true;
            }
        }
    }

    unique_ptr<FunctionData> ReadArrayBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names)
    {
        // TODO: process input tile coordinates
        string arrayname = StringValue::Get(input.inputs[0]);
        unique_ptr<ArrayReadData> bind_data = make_uniq<ArrayReadData>(arrayname);

        if (bind_data->dim_len == 1)
        {
            return_types.push_back(LogicalType::UINTEGER);
            return_types.push_back(LogicalType::DOUBLE);

            names.emplace_back("idx");
            names.emplace_back("val");
        }
        else if (bind_data->dim_len == 2)
        {
            return_types.push_back(LogicalType::UINTEGER);
            return_types.push_back(LogicalType::UINTEGER);
            return_types.push_back(LogicalType::DOUBLE);

            names.emplace_back("x");
            names.emplace_back("y");
            names.emplace_back("val");
        }
        else
        {
            throw NotImplementedException("Only 1D and 2D arrays are supported");
        }

        return std::move(bind_data);
    }

    unique_ptr<GlobalTableFunctionState> ReadArrayGlobalStateInit(ClientContext &context, TableFunctionInitInput &input)
    {
        return std::move(make_uniq<ArrayReadGlobalState>(context, input));
    }

    unique_ptr<LocalTableFunctionState> ReadArrayLocalStateInit(ExecutionContext &context, TableFunctionInitInput &input,
                                                                GlobalTableFunctionState *gstate)
    {
        return std::move(make_uniq<ArrayReadLocalState>(context.client, input, gstate->Cast<ArrayReadGlobalState>()));
    }

    // void ReadArrayComplexFilterPushdown(
    //     ClientContext &context, LogicalGet &get, FunctionData *bind_data_p, vector<unique_ptr<Expression>> &filters)
    // {
    //     std::cerr << "[" << pthread_self()
    //               << "]"
    //                  "CSVComplexFilterPushdown"
    //               << std::endl;
    // }

    TableFunction ArrayExtension::GetTableFunction()
    {
        TableFunction function = TableFunction(
            "read_array", {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::INTEGER)},
            ReadArrayFunction, ReadArrayBind, ReadArrayGlobalStateInit, ReadArrayLocalStateInit);
        // function.filter_pushdown = true;
        function.projection_pushdown = true;
        function.filter_prune = true;

        // function.pushdown_complex_filter = ReadArrayComplexFilterPushdown;
        // TODO: table_function.function_info = std::move(function_info);

        return function;
    }
}