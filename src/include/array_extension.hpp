#pragma once

#include "duckdb.hpp"

namespace duckdb {

class ArrayExtension : public Extension {
   public:
    void Load(DuckDB &db) override;
    std::string Name() override;

    static TableFunction GetReadArrayFunction();
    static CopyFunction GetCopyFunction();
    static ScalarFunction GetCreateArrayFunction();

    static void ResetPVBufferStats();
    static void PrintPVBufferStats();
};

}  // namespace duckdb
