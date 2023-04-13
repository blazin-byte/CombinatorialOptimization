import numpy as np
import mysql.connector
import asyncio
import time
import pandas as pd
from mysql.connector import errorcode
import pymysql


# Use this a different file to actually querry from the databases and construct table
db1_config = {
    "host": "simflightprices.c2cptuppgf0q.us-east-2.rds.amazonaws.com",
    "user": "admin",
    "password": "12345678",
    "database": "simflightprices"
}

db2_config = {
    "host": "simhotelprices.c2cptuppgf0q.us-east-2.rds.amazonaws.com",
    "user": "admin",
    "password": "12345678",
    "database": "simhotelprices"
}


def generate_time_series(num_signals=100, signal_length=7):
    """
    Function that takes the number of num_signals (default = 6) and signal_length (default = 100) and generate num_signals hotel price signals and 2*num_signals flight prices for outgoing and returns the list of all all flight price signals and hotel price signals
    """
    flights = np.random.rand(num_signals*2, signal_length)*1000
    flights = [[float(val) for val in row] for row in flights]
    hotels = np.random.rand(num_signals, signal_length)*200
    hotels = [[float(val) for val in row] for row in hotels]
    return flights, hotels


def generate_time_series_2D(num_signals=100, signal_length=7):
    """
    Function that takes the number of num_signals (default = 6) and signal_length (default = 100) and generate num_signals hotel price signals and 2*num_signals flight prices for outgoing and returns the list of all all flight price signals and hotel price signals
    """
    flights = np.random.rand(num_signals*6, signal_length)*1000
    flights = [[float(val) for val in row] for row in flights]
    hotels = np.random.rand(num_signals*2, signal_length)*200
    hotels = [[float(val) for val in row] for row in hotels]
    return flights, hotels


# Upload Flight Data to FLight Database and Hotel Data to Hotel Database
def upload_to_databases(flights, hotels):
    """
    This function takes two lists of lists of flights and hotel prices and uploads these series into two different AWS MySQL databases
    """
    # Connect to the first database
    cnx1 = mysql.connector.connect(**db1_config)
    cursor1 = cnx1.cursor()

    # Connect to the second database
    cnx2 = mysql.connector.connect(**db2_config)
    cursor2 = cnx2.cursor()

    # # Checking to see if simflightprices database exists
    # cursor1.execute("SHOW DATABASES")
    # databases = cursor1.fetchall()
    # if (bytearray(b'simflightprices'),) not in databases:
    #     print("The database 'simflightprices' does not exist")
    #     cursor1.execute("CREATE DATABASE simflightprices")
    #     cursor1.execute("USE simflightprices")
    #     print("The database 'simflightprices' was created and is now in use")
    # else:
    #     # The database exists, so continue with your program
    #     cursor1.execute("USE simflightprices")

    # # Checking to see if simhotelprices database exists
    # cursor2.execute("SHOW DATABASES")
    # databases = cursor2.fetchall()
    # if (bytearray(b'simhotelprices'),) not in databases:
    #     print("The database 'simhotelprices' does not exist")
    #     cursor2.execute("CREATE DATABASE simhotelprices")
    #     cursor2.execute("USE simhotelprices")
    #     print("The database 'simhotelprices' was created and is now in use")
    # else:
    #     # The database exists, so continue with your program
    #     cursor2.execute("USE simhotelprices")

    # Adding Table in Flights Database
    cursor1.execute("DROP TABLE IF EXISTS Price_Table")
    cursor1.execute("CREATE TABLE Price_Table (flight VARCHAR(20), "
                    + ", ".join(["day_" + str(i) +
                                 " FLOAT" for i in range(len(flights[0]))])
                    + ")")

    # Insert data into table
    rows = [("flight" + str(i),) + tuple(row)
            for i, row in enumerate(flights, 1)]
    insert_query = "INSERT INTO Price_Table (flight, " + ", ".join(["day_" + str(i) for i in range(len(flights[0]))]) + ") " \
        + "VALUES (" + ", ".join(["%s" for i in range(len(flights[0]) + 1)]) + ")"
    cursor1.executemany(insert_query, rows)
    cnx1.commit()

    # Close connection
    cursor1.close()
    cnx1.close()

# Adding Table in Hotels Database
    cursor2.execute("DROP TABLE IF EXISTS Price_Table")
    cursor2.execute("CREATE TABLE Price_Table (hotel VARCHAR(20), "
                    + ", ".join(["day_" + str(i) +
                                 " FLOAT" for i in range(len(hotels[0]))])
                    + ")")

    # Insert data into table
    rows = [("hotel" + str(i),) + tuple(row)
            for i, row in enumerate(hotels, 1)]
    insert_query = "INSERT INTO Price_Table (hotel, " + ", ".join(["day_" + str(i) for i in range(len(hotels[0]))]) + ") " \
        + "VALUES (" + ", ".join(["%s" for i in range(len(hotels[0]) + 1)]) + ")"
    cursor2.executemany(insert_query, rows)
    cnx2.commit()

    # Close connection
    cursor1.close()
    cnx1.close()


# Synchronous Fetching of Data
def fetch_data():
    start_time = time.time()
    query = "SELECT * FROM Price_Table"
    # Connect to MySQL database
    cnx1 = mysql.connector.connect(**db1_config)
    # Execute query and retrieve data as pandas dataframe
    df1 = pd.read_sql_query(query, cnx1)
    df1 = df1.rename(columns={'flight': 'type'})

    # Close database connection
    cnx1.close()
    # Connect to MySQL database
    cnx2 = mysql.connector.connect(**db2_config)
    # Execute query and retrieve data as pandas dataframe
    df2 = pd.read_sql_query(query, cnx2)
    df2 = df2.rename(columns={'hotel': 'type'})
    # Close database connection
    cnx2.close()
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    end_time = time.time()
    query_time = end_time - start_time
    #print(f"Query executed in {query_time:.2f} seconds.")
    return df, query_time


# async def fetch_from_database(database_config, query):
#     try:
#         cnx = mysql.connector.connect(**database_config)
#         cursor = cnx.cursor()
#         await cursor.execute(query)
#         result = cursor.fetchall()
#         column_names = cursor.column_names
#         cursor.close()
#         cnx.close()
#         return pd.DataFrame(result, columns=column_names)
#     except mysql.connector.Error as err:
#         print(f"Error accessing database: {err}")
#         return None


# async def fetch_from_databases(db1_config, db2_config, query):
#     db1_task = asyncio.create_task(fetch_from_database(db1_config, query))
#     db2_task = asyncio.create_task(fetch_from_database(db2_config, query))
#     results = await asyncio.gather(db1_task, db2_task)
#     return pd.concat(results)


# def query_and_store():
#     query = "SELECT * FROM Price_Table"
#     table_name = "combined_table"
#     start_time = time.time()
#     loop = asyncio.get_event_loop()

#     try:
#         df = loop.run_until_complete(
#             fetch_from_databases(db1_config, db2_config, query))
#         df.to_csv(f"{table_name}.csv", index=False)
#         elapsed_time = time.time() - start_time
#         print(f"Elapsed time: {elapsed_time} seconds")
#     except RuntimeError as e:
#         if str(e) != "Event loop is closed":
#             raise e
#     finally:
#         if loop.is_running():
#             loop.stop()
#             loop.run_forever()
#         loop.close()


# async def fetch_from_database(database_config, query):
#     try:
#         cnx = mysql.connector.connect(**database_config)
#         cursor = cnx.cursor()
#         await cursor.execute(query)
#         result = cursor.fetchall()
#         column_names = cursor.column_names
#         cursor.close()
#         cnx.close()
#         return pd.DataFrame(result, columns=column_names)
#     except mysql.connector.Error as err:
#         print(f"Error accessing database: {err}")
#         return None


# async def fetch_from_databases(db1_config, db2_config, query):
#     db1_task = asyncio.create_task(fetch_from_database(db1_config, query))
#     db2_task = asyncio.create_task(fetch_from_database(db2_config, query))
#     results = await asyncio.gather(db1_task, db2_task)
#     return pd.concat(results)


# def query_and_store():
#     query = "SELECT * FROM my_table"
#     table_name = "combined_table"
#     start_time = time.time()
#     loop = asyncio.get_event_loop()
#     df = loop.run_until_complete(
#         fetch_from_databases(db1_config, db2_config, query))
#     df.to_csv(f"{table_name}.csv", index=False)
#     elapsed_time = time.time() - start_time
#     print(f"Elapsed time: {elapsed_time} seconds")


# # Function to delete all data from a MySql Database
# def clear_database(database_config):
#     try:
#         # Connect to the database
#         conn = mysql.connector.connect(**database_config)

#         # Create a cursor object
#         cursor = conn.cursor()

#         # Loop through all tables in the database and truncate them
#         cursor.execute("SHOW TABLES")
#         tables = cursor.fetchall()
#         for table in tables:
#             table_name = table[0]
#             cursor.execute(f"TRUNCATE TABLE {table_name}")

#         # Commit the changes
#         conn.commit()

#         print("Successfully cleared all data from the database.")

#     except Exception as e:
#         print(f"Error: {e}")

#     finally:
#         # Close the database connection
#         if conn.is_connected():
#             cursor.close()
#             conn.close()
