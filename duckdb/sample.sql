INSTALL nanodbc FROM community;
LOAD nanodbc;

select extension_name, version from duckdb_extensions();