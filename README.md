# Array

## TODO

- `read_array()` multi attributes for dense array reading

## Function Descriptions

### Read Function

- Only COO array supports multi-attributes currently.

```SQL
-- Read entire 3D array
SELECT x, y, z, a0 FROM read_array('finedust');

-- Read (0, 0) tile of 2D array 
SELECT x, y, a0 FROM read_array('finedust', coords=[0, 0]);

-- Read (0, 0) tile of 2D array with three attributes
SELECT x, y, a0, a1, a2 FROM read_array('finedust', coords=[0, 0]);
```

### COPY Function

- `MODE 0`: COO table with only dimensions and value to a dense array
- `MODE 1`: table with only values (sorted) to dense array
    - The table must be filled
    - `COORD_X`, `COORD_Y`, and `COORD_Z` are required
- `MODE 2`: COO table with other attributes to COO array
    - The first `d` columns are assumed to be attributes for dimensions
- `MODE 3`: COO table with other attributes to dense array

```sql
COPY (
    SELECT timestamp::UINTEGER, latitude::UINTEGER, longitude::UINTEGER, pm10::DOUBLE 
    FROM Finedust_idx 
    ORDER BY timestamp ASC, longitude ASC, latitude ASC) 
TO 'finedust_pm10.tilestore'(MODE 0, COORD_X 0, COORD_Y 0, COORD_Z 0);
```