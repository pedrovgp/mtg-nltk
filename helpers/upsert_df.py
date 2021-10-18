# Upsert function for pandas to_sql with postgres
# https://stackoverflow.com/questions/1109061/insert-on-duplicate-update-in-postgresql/8702291#8702291
# https://www.postgresql.org/docs/devel/sql-insert.html#SQL-ON-CONFLICT
import pandas as pd
import sqlalchemy
import uuid
import os


def upsert_df(df: pd.DataFrame, table_name: str, engine: sqlalchemy.engine.Engine):
    """Implements the equivalent of pd.DataFrame.to_sql(..., if_exists='update')
    (which does not exist). Creates or updates the db records based on the
    dataframe records.
    Conflicts to determine update are based on the dataframes index.
    This will set primary keys on the table equal to the index names

    1. Create a temp table from the dataframe
    2. Insert/update from temp table into table_name

    Returns: True if successful

    """

    # If the table does not exist, we should just use to_sql to create it
    if not engine.execute(
        f"""SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE  table_schema = 'public'
            AND    table_name   = '{table_name}');
            """
    ).first()[0]:
        df.to_sql(table_name, engine)
        return True

    # If it already exists...
    temp_table_name = f"temp_{uuid.uuid4().hex[:6]}"
    df.to_sql(temp_table_name, engine, index=True)

    index = list(df.index.names)
    index_sql_txt = ", ".join([f'"{i}"' for i in index])
    columns = list(df.columns)
    headers = index + columns
    headers_sql_txt = ", ".join(
        [f'"{i}"' for i in headers]
    )  # index1, index2, ..., column 1, col2, ...

    # col1 = exluded.col1, col2=excluded.col2
    update_column_stmt = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in columns])

    # For the ON CONFLICT clause, postgres requires that the columns have unique constraint
    query_pk = f"""
    ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS unique_constraint_for_upsert;
    ALTER TABLE "{table_name}" ADD CONSTRAINT unique_constraint_for_upsert UNIQUE ({index_sql_txt});
    """
    engine.execute(query_pk)

    # Compose and execute upsert query
    query_upsert = f"""
    INSERT INTO "{table_name}" ({headers_sql_txt}) 
    SELECT {headers_sql_txt} FROM "{temp_table_name}"
    ON CONFLICT ({index_sql_txt}) DO UPDATE 
    SET {update_column_stmt};
    """
    engine.execute(query_upsert)
    engine.execute(f'DROP TABLE "{temp_table_name}"')

    return True


if __name__ == "__main__":
    # TESTS (create environment variable DB_STR to do it)
    engine = sqlalchemy.create_engine(os.getenv("DB_STR"))

    indexes = ["id1", "id2"]

    df = pd.DataFrame(
        {
            "id1": [1, 2, 3, 3],
            "id2": ["a", "a", "b", "c"],
            "name": ["name1", "name2", "name3", "name4"],
            "age": [20, 32, 29, 68],
        }
    ).set_index(indexes)

    df_update = pd.DataFrame(
        {
            "id1": [1, 2, 3],
            "id2": ["a", "a", "b"],
            "name": ["surname1", "surname2", "surname3"],
            "age": [13, 44, 29],
        }
    ).set_index(indexes)

    df_insert = pd.DataFrame(
        {
            "id1": [1],
            "id2": ["d"],
            "name": ["dname"],
            "age": [100],
        }
    ).set_index(indexes)

    expected_result = (
        pd.DataFrame(
            {
                "id1": [1, 2, 3, 3, 1],
                "id2": ["a", "a", "b", "c", "d"],
                "name": ["surname1", "surname2", "surname3", "name4", "dname"],
                "age": [13, 44, 29, 68, 100],
            }
        )
        .set_index(indexes)
        .sort_index()
    )

    TNAME = "test_upsert_df"
    upsert_df(df=df, table_name=TNAME, engine=engine)
    upsert_df(df=df_update, table_name=TNAME, engine=engine)
    upsert_df(df=df_insert, table_name=TNAME, engine=engine)
    result = pd.read_sql_table(TNAME, engine).set_index(indexes).sort_index()
    assert (result == expected_result).all().all()
    print("Passed tests")
