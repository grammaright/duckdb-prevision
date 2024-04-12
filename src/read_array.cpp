
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
        string arrayname;
    };

    struct ArrayReadGlobalState : public GlobalTableFunctionState
    {
    public:
        ArrayReadGlobalState(ClientContext &context, TableFunctionInitInput &input)
        {
            auto &data = input.bind_data->Cast<ArrayReadData>();

            uint64_t **_dim_domains;
            uint64_t *_tile_size;
            uint64_t *_array_size_in_tile;
            storage_util_get_dim_domains(data.arrayname.c_str(), &_dim_domains, &dim_len);
            storage_util_get_tile_extents(data.arrayname.c_str(), &_tile_size, &dim_len);
            storage_util_get_dcoord_lens(_dim_domains, _tile_size, dim_len, &_array_size_in_tile);

            array_size_in_tile = vector<uint64_t>(_array_size_in_tile, _array_size_in_tile + dim_len);
            tile_size = vector<uint64_t>(_tile_size, _tile_size + dim_len);

            storage_util_free_dim_domains(&_dim_domains, dim_len);
            storage_util_free_tile_extents(&_tile_size, dim_len);
            storage_util_free_dcoord_lens(&_array_size_in_tile);

            current_coords_in_tile = vector<uint64_t>(dim_len, 0);
            finished = false;

            // copy
            column_ids.assign(input.column_ids.begin(), input.column_ids.end());
            projection_ids.assign(input.projection_ids.begin(), input.projection_ids.end());
        };

        ~ArrayReadGlobalState(){};

    public:
        // tile related:
        vector<uint64_t> array_size_in_tile;
        vector<uint64_t> tile_size;
        uint32_t dim_len;

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
        for (uint32_t idx = 0; idx < gstate.dim_len; idx++)
        {
            dcoords[idx] = gstate.current_coords_in_tile[idx];
        }
        auto arrname = data.arrayname.c_str();
        // why allocate newly?
        char *arrname_char = new char[1024];
        strcpy(arrname_char, arrname);

        // TODO: Consider sparse tile in the future
        array_key key = {arrname_char, "a", dcoords.get(), 2, BF_EMPTYTILE_DENSE};

        PFpage *page;
        BF_GetBuf(key, &page);

        auto size = page->pagebuf_len / sizeof(double);
        double *pageval = (double *)bf_util_get_pagebuf(page);

        if (gstate.projection_ids.size() > 0)
        {
            // filter_prune ON
            for (uint64_t idx = 0; idx < size; idx++)
            {
                uint64_t *coords;
                bf_util_calculate_nd_from_1d_row_major(idx, bf_util_get_tile_extents(page), 2, &coords);

                double val = pageval[idx];

                for (uint32_t i = 0; i < gstate.projection_ids.size(); i++)
                {
                    auto dest = gstate.column_ids[gstate.projection_ids[i]];
                    if (dest == 0)
                    {
                        FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[0];
                    }
                    else if (dest == 1)
                    {
                        FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[1];
                    }
                    else if (dest == 2)
                    {
                        FlatVector::GetData<double>(output.data[i])[idx] = val;
                    }
                }

                free(coords);
            }
        }
        else if (gstate.column_ids.size() == output.data.size())
        {
            // no filter prune
            for (uint64_t idx = 0; idx < size; idx++)
            {
                uint64_t *coords;
                bf_util_calculate_nd_from_1d_row_major(idx, bf_util_get_tile_extents(page), 2, &coords);

                double val = pageval[idx];

                for (uint32_t i = 0; i < gstate.column_ids.size(); i++)
                {
                    if (gstate.column_ids[i] == 0)
                    {
                        FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[0];
                    }
                    else if (gstate.column_ids[i] == 1)
                    {
                        FlatVector::GetData<uint32_t>(output.data[i])[idx] = coords[1];
                    }
                    else if (gstate.column_ids[i] == 2)
                    {
                        FlatVector::GetData<double>(output.data[i])[idx] = val;
                    }
                }

                free(coords);
            }
        }
        else
        {
            // projection_pushdown and filter_prune are both false
            auto xs = FlatVector::GetData<uint32_t>(output.data[0]);
            auto ys = FlatVector::GetData<uint32_t>(output.data[1]);
            auto vals = FlatVector::GetData<double>(output.data[2]); // double type assumed

            for (uint64_t idx = 0; idx < size; idx++)
            {
                uint64_t *coords;
                bf_util_calculate_nd_from_1d_row_major(idx, bf_util_get_tile_extents(page), 2, &coords);

                double val = pageval[idx];

                xs[idx] = coords[0];
                ys[idx] = coords[1];
                vals[idx] = val;

                free(coords);
            }
        }

        output.SetCardinality(size);
        output.Verify();

        BF_UnpinBuf(key);
        delete arrname_char;

        // move the current_coords_in_tile to the next tile
        for (int64_t idx = gstate.dim_len - 1; idx >= 0; idx--)
        {
            gstate.current_coords_in_tile[idx] += 1;
            if (gstate.current_coords_in_tile[idx] < gstate.array_size_in_tile[idx])
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
        auto bind_data = make_uniq<ArrayReadData>();
        bind_data->arrayname = StringValue::Get(input.inputs[0]);

        return_types.push_back(LogicalType::UINTEGER);
        return_types.push_back(LogicalType::UINTEGER);
        return_types.push_back(LogicalType::DOUBLE);

        names.emplace_back("x");
        names.emplace_back("y");
        names.emplace_back("val");

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