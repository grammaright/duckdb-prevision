# Array

## Function Descriptions

### Read Function

```SQL
-- Read entire 3D array
SELECT x, y, z, val FROM read_array('finedust');

-- Read (0, 0) tile of 2D array 
SELECT x, y, val FROM read_array('finedust', coords=[0, 0]);

-- Read (0, 0) tile of 2D COO array (i.e., dim[0] = x,y,val and dim[1] = rows)
SELECT x, y, val FROM read_array('finedust', coords=[0, 0] array_type=\"COO\");
```

### COPY Function

- `MODE 0`: COO to Array
- `MODE 1`: Dense array (only values) to array
    - `COORD_X`, `COORD_Y`, and `COORD_Z` are required

```sql
COPY (
    SELECT timestamp::UINTEGER, latitude::UINTEGER, longitude::UINTEGER, pm10::DOUBLE 
    FROM Finedust_idx 
    ORDER BY timestamp ASC, longitude ASC, latitude ASC) 
TO 'finedust_pm10.tilestore'(MODE 0, COORD_X 0, COORD_Y 0, COORD_Z 0);
```