
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include "duckdb/common/bind_helpers.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/serializer/write_stream.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/operator/csv_scanner/sniffer/csv_sniffer.hpp"

#include "duckdb/function/scalar/string_functions.hpp"
#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"

#include <limits>

#include "array_extension.hpp"

extern "C"
{
#include "bf.h"
}

namespace duckdb
{
    class WriteArrayData : public FunctionData
    {
    public:
        WriteArrayData(string file_path, vector<uint32_t> tile_coords)
            : tile_coords(tile_coords)
        {
            array_name = file_path.substr(0, file_path.find_last_of("."));

            uint64_t **_dim_domains;
            uint64_t *_tile_size;
            uint64_t *_array_size_in_tile;
            storage_util_get_dim_domains(array_name.c_str(), &_dim_domains, &dim_len);
            storage_util_get_tile_extents(array_name.c_str(), &_tile_size, &dim_len);
            storage_util_get_dcoord_lens(_dim_domains, _tile_size, dim_len, &_array_size_in_tile);

            array_size_in_tile = vector<uint64_t>(_array_size_in_tile, _array_size_in_tile + dim_len);
            tile_size = vector<uint64_t>(_tile_size, _tile_size + dim_len);

            storage_util_free_dim_domains(&_dim_domains, dim_len);
            storage_util_free_tile_extents(&_tile_size, dim_len);
            storage_util_free_dcoord_lens(&_array_size_in_tile);
        }

        unique_ptr<FunctionData> Copy() const override;
        bool Equals(const FunctionData &other) const override;

        string array_name;
        vector<uint32_t> tile_coords;

        uint32_t dim_len;

        vector<uint64_t> array_size_in_tile;
        vector<uint64_t> tile_size;
    };

    unique_ptr<FunctionData> WriteArrayData::Copy() const
    {
        return nullptr;
    }

    bool WriteArrayData::Equals(const FunctionData &other_p) const
    {
        return false;
    }

    struct LocalWriteArrayData : public LocalFunctionData
    {
    };

    struct GlobalWriteArrayData : public GlobalFunctionData
    {
    public:
        GlobalWriteArrayData(ClientContext &context, FunctionData &bind_data,
                             const string &file_path)
        {
            auto &data = bind_data.Cast<WriteArrayData>();

            // getbuffer
            auto dcoords = make_uniq_array<uint64_t>(2);
            dcoords[0] = data.tile_coords[0];
            dcoords[1] = data.tile_coords[1];

            auto arrname = data.array_name.c_str();
            // why allocate newly?
            arrname_char = new char[1024];
            strcpy(arrname_char, arrname);

            // TODO: Consider sparse tile in the future
            key = {arrname_char, "a", dcoords.get(), data.dim_len, BF_EMPTYTILE_DENSE};

            PFpage *page;
            BF_GetBuf(key, &page);

            buf_size = page->pagebuf_len / sizeof(double);
            buf = (double *)bf_util_get_pagebuf(page);
        }

        ~GlobalWriteArrayData()
        {
            // FIXME: why double free?
            // BF_UnpinBuf(key);
            // delete arrname_char;
        }

        void unpin()
        {
            BF_UnpinBuf(key);
            delete arrname_char;
        }

        uint64_t buf_size;
        double *buf;

        uint64_t cur_idx = 0;

    private:
        array_key key;
        char *arrname_char;

        // vector<uint64_t> current_coords_in_tile;
        // bool finished;
    };

    static unique_ptr<FunctionData>
    WriteArrayBind(ClientContext &context, CopyFunctionBindInput &input,
                   const vector<string> &names, const vector<LogicalType> &sql_types)
    {
        uint32_t x, y;

        // check all the options in the copy info
        for (auto &option : input.info.options)
        {
            if (option.first == "COORD_X")
            {
                auto incoords = option.second;
                for (auto incoord : incoords)
                {
                    auto val = incoord.GetValue<uint32_t>();
                    x = val;
                    break; // I don't know why it gives a vector
                }
            }
            else if (option.first == "COORD_Y")
            {
                auto incoords = option.second;
                for (auto incoord : incoords)
                {
                    auto val = incoord.GetValue<uint32_t>();
                    y = val;
                    break; // I don't know why it gives a vector
                }
            }
        }

        auto file_path = input.info.file_path;
        vector<uint32_t> tile_coords = {x, y};
        auto bind_data = make_uniq<WriteArrayData>(file_path, tile_coords);
        return std::move(bind_data);
    }

    static unique_ptr<LocalFunctionData> WriteArrayInitializeLocal(ExecutionContext &context, FunctionData &bind_data)
    {
        return std::move(make_uniq<LocalWriteArrayData>());
    }

    static unique_ptr<GlobalFunctionData> WriteArrayInitializeGlobal(ClientContext &context, FunctionData &bind_data,
                                                                     const string &file_path)
    {
        return std::move(make_uniq<GlobalWriteArrayData>(context, bind_data, file_path));
    }

    static void WriteArraySink(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                               LocalFunctionData &lstate, DataChunk &input)
    {
        // NOTE: I assume that only one thread runs
        auto array_gstate = gstate.Cast<GlobalWriteArrayData>();

        // We don't know what vector type DuckDB will give
        // So we need to convert it to unified vector format
        // vector type ref: https://youtu.be/bZOvAKGkzpQ?si=ShnWtUDKNIm7ymo8&t=1265
        input.data[0].Flatten(input.size()); // FIXME: Maybe performance panalty. exploit the vector type
        auto vector = FlatVector::GetData<double>(input.data[0]);

        D_ASSERT(array_gstate.cur_idx + input.size() <= array_gstate.buf_size);

        for (idx_t i = 0; i < input.size(); i++)
        {
            array_gstate.buf[array_gstate.cur_idx] = vector[i];
            array_gstate.cur_idx++;
        }
    }

    //===--------------------------------------------------------------------===//
    // Combine
    //===--------------------------------------------------------------------===//
    static void WriteArrayCombine(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                                  LocalFunctionData &lstate)
    {
    }

    //===--------------------------------------------------------------------===//
    // Finalize
    //===--------------------------------------------------------------------===//
    void WriteArrayFinalize(ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate)
    {
        auto array_gstate = gstate.Cast<GlobalWriteArrayData>();
        array_gstate.unpin();
    }

    //===--------------------------------------------------------------------===//
    // Execution Mode
    //===--------------------------------------------------------------------===//
    CopyFunctionExecutionMode WriteArrayExecutionMode(bool preserve_insertion_order, bool supports_batch_index)
    {
        return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
    }
    //===--------------------------------------------------------------------===//
    // Prepare Batch
    //===--------------------------------------------------------------------===//
    struct WriteArrayBatchData : public PreparedBatchData
    {
    };

    unique_ptr<PreparedBatchData> WriteArrayPrepareBatch(ClientContext &context, FunctionData &bind_data,
                                                         GlobalFunctionData &gstate,
                                                         unique_ptr<ColumnDataCollection> collection)
    {
        return std::move(make_uniq<WriteArrayBatchData>());
    }

    //===--------------------------------------------------------------------===//
    // Flush Batch
    //===--------------------------------------------------------------------===//
    void WriteArrayFlushBatch(ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                              PreparedBatchData &batch)
    {
    }

    idx_t WriteArrayFileSize(GlobalFunctionData &gstate)
    {
        return 1;
    }

    CopyFunction ArrayExtension::GetCopyFunction()
    {
        CopyFunction info("tilestore");
        info.copy_to_bind = WriteArrayBind;
        info.copy_to_initialize_local = WriteArrayInitializeLocal;
        info.copy_to_initialize_global = WriteArrayInitializeGlobal;
        info.copy_to_sink = WriteArraySink;
        info.copy_to_combine = WriteArrayCombine;
        info.copy_to_finalize = WriteArrayFinalize;
        info.execution_mode = WriteArrayExecutionMode;
        // info.prepare_batch = WriteArrayPrepareBatch;
        // info.flush_batch = WriteArrayFlushBatch;
        info.file_size_bytes = WriteArrayFileSize;

        info.plan = nullptr;

        info.copy_from_bind = nullptr;
        info.copy_from_function = ArrayExtension::GetTableFunction();

        info.extension = "tilestore";

        return info;
    }

}