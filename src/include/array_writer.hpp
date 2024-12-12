#pragma once

#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <limits>

#include "array_extension.hpp"
#include "copy_array.hpp"

#include "buffer/bf.h"

namespace duckdb {

class CopyArrayWriter {
   public:
    virtual void WriteArrayData(ExecutionContext &context,
                                FunctionData &bind_data,
                                GlobalFunctionData &gstate,
                                LocalFunctionData &lstate,
                                DataChunk &input) = 0;
};

class CopyArrayData : public FunctionData {
   public:
    CopyArrayData(string file_path, ArrayCopyFunctionExecutionMode mode);

    unique_ptr<FunctionData> Copy() const override;
    bool Equals(const FunctionData &other) const override;

   public:
    ArrayCopyFunctionExecutionMode mode;
    string array_name;

    uint32_t dim_len;
    vector<uint64_t> array_size_in_tile;
    vector<uint64_t> tile_size;
};

class GlobalCOOToArrayWriteArrayData : public GlobalWriteArrayData {
   public:
    GlobalCOOToArrayWriteArrayData(ClientContext &context,
                                   FunctionData &bind_data,
                                   const string &file_path);
};

class COOToArrayCopyArrayWriter : public CopyArrayWriter {
   public:
    COOToArrayCopyArrayWriter();
    void WriteArrayData(ExecutionContext &context, FunctionData &bind_data,
                        GlobalFunctionData &gstate, LocalFunctionData &lstate,
                        DataChunk &input) override;
};

class GlobalDenseToTileWriteArrayData : public GlobalWriteArrayData {
   public:
    GlobalDenseToTileWriteArrayData(ClientContext &context,
                                    FunctionData &bind_data,
                                    const string &file_path);
};

class DenseToTileCopyArrayData : public CopyArrayData {
   public:
    DenseToTileCopyArrayData(string file_path,
                             ArrayCopyFunctionExecutionMode mode,
                             vector<uint64_t> tile_coords);

   public:
    vector<uint64_t> tile_coords;
};

class DenseToTileCopyArrayWriter : public CopyArrayWriter {
   public:
    DenseToTileCopyArrayWriter(GlobalFunctionData &gstate,
                               DenseToTileCopyArrayData &array_data);
    void WriteArrayData(ExecutionContext &context, FunctionData &bind_data,
                        GlobalFunctionData &gstate, LocalFunctionData &lstate,
                        DataChunk &input) override;
    uint64_t cur_idx;
};

class GlobalCoomaToDenseWriteArrayData : public GlobalWriteArrayData {
   public:
    GlobalCoomaToDenseWriteArrayData(ClientContext &context,
                                     FunctionData &bind_data,
                                     const string &file_path);

   private:
    bool isFirst;
    int dataLen;

    friend class CoomaToDenseCopyArrayWriter;
};

class CoomaToDenseCopyArrayWriter : public CopyArrayWriter {
   public:
    CoomaToDenseCopyArrayWriter(GlobalFunctionData &gstate);

    void WriteArrayData(ExecutionContext &context, FunctionData &bind_data,
                        GlobalFunctionData &gstate, LocalFunctionData &lstate,
                        DataChunk &input) override;
};

class GlobalCoomaToCooWriteArrayData : public GlobalWriteArrayData {
   public:
    GlobalCoomaToCooWriteArrayData(ClientContext &context,
                                   FunctionData &bind_data,
                                   const string &file_path);

   private:
    bool isFirst;
    int dataLen;

    friend class CoomaToCooCopyArrayWriter;
};

class CoomaToCooCopyArrayWriter : public CopyArrayWriter {
   public:
    CoomaToCooCopyArrayWriter(GlobalFunctionData &gstate);

    void WriteArrayData(ExecutionContext &context, FunctionData &bind_data,
                        GlobalFunctionData &gstate, LocalFunctionData &lstate,
                        DataChunk &input) override;
};

}  // namespace duckdb