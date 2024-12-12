#include "array_extension.hpp"
#include "array_writer.hpp"

#include "buffer/bf.h"

namespace duckdb {

CopyArrayData::CopyArrayData(string file_path,
                             ArrayCopyFunctionExecutionMode mode) : mode(mode) {
    array_name = file_path.substr(0, file_path.find_last_of("."));

    uint64_t **_dim_domains;
    uint64_t *_tile_size;
    uint64_t *_array_size_in_tile;
    storage_util_get_dim_domains(array_name.c_str(), &_dim_domains, &dim_len);
    storage_util_get_tile_extents(array_name.c_str(), &_tile_size, &dim_len);
    storage_util_get_dcoord_lens(_dim_domains, _tile_size, dim_len,
                                 &_array_size_in_tile);

    array_size_in_tile =
        vector<uint64_t>(_array_size_in_tile, _array_size_in_tile + dim_len);
    tile_size = vector<uint64_t>(_tile_size, _tile_size + dim_len);

    storage_util_free_dim_domains(&_dim_domains, dim_len);
    storage_util_free_tile_extents(&_tile_size, dim_len);
    storage_util_free_dcoord_lens(&_array_size_in_tile);
}

unique_ptr<FunctionData> CopyArrayData::Copy() const { return nullptr; }

bool CopyArrayData::Equals(const FunctionData &other_p) const { return false; }

}  // namespace duckdb