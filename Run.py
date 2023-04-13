
# %%
#Imports and Initialization
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import AlgorithmFxns as algs
import PriceSimFxns as psim

import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

num_options = 3
num_days = 4
num_stay = num_days//2

flights, hotels = psim.generate_time_series_2D(
    num_signals=num_options, signal_length=num_days)
psim.upload_to_databases(flights=flights, hotels=hotels)
df = psim.fetch_data()[0]
df = algs.process_df_2D(df, num_options=num_options)


# %%
# Creating Graph
G = algs.generate_graph(df, num_days, num_stay, num_options)
pos = graphviz_layout(G, prog="dot")
# draw the nodes and edges
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edges(G, pos)
# draw the edge weights
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# show the plot
plt.axis('off')
plt.show()


# Ok Now What??


# %%
# Generate and Upload Data

#
# Should only have one iteration and basically do exactly what above is doing for POC
# num_options_list = []
# brute_force_1D_list = []
# proposed_alg_1D_list = []
# for i in range(1, 11):
#     flights, hotels = psim.generate_time_series(num_options*i*2, num_days)
#     psim.upload_to_databases(flights=flights, hotels=hotels)
#     df = psim.fetch_data()[0]
#     df = algs.process_df(df=df, num_options=num_options*i)
#     num_options_list.append(num_options*i*10)
#     proposed_alg_1D_list.append(algs.proposedAlg1D(df, num_stay)[2])
#     brute_force_1D_list.append(algs.bruteforce_1D(df, num_stay)[2])
#     print("Iteration " + str(i))


# Plotting both the curves simultaneously
# plt.plot(num_options_list, brute_force_1D_list,
#          color='r', label='Brute Force Alg')
# plt.plot(num_options_list, proposed_alg_1D_list,
#          color='g', label='Proposed Alg')
# plt.legend(['Brute Force Alg', 'Proposed Alg'])
# # Naming the x-axis, y-axis and the whole graph
# plt.xlabel("Number of Options for Flights(Depart),Hotels, and Flights(Return)")
# plt.ylabel("Time (secs)")
# plt.title("Comparison of Run-Time between Proposed Alg and Brute Force Alg")


# TODO: Use nx.draw to print the finished graph...
# %%
# TROUBLESHOOT SPACE


# %%
# ASYNC TROUBLESHOOT
#                     # For some reason, there seems to be an error running the event loop...

#                     # For right now, focus on gathering data to present...

# async def get_data(db_config):
#     query = 'SELECT * FROM Price_Table;'
#     async with aiomysql.connect(**db_config) as conn:
#         async with conn.cursor() as cur:
#             await cur.execute(query)
#             rows = await cur.fetchall()
#             df = pd.DataFrame(rows, columns=[col[0]
#                                              for col in cur.description])
#     return df

# async def main():
#     start_time = time.time()
#     task1 = asyncio.create_task(get_data(db1_config))
#     task2 = asyncio.create_task(get_data(db2_config))
#     await asyncio.gather(task1, task2)
#     df1 = await task1
#     df2 = await task2
#     df_concat = pd.concat([df1, df2])
#     end_time = time.time()
#     query_time = end_time - start_time
#     print(f"Query executed in {query_time:.2f} seconds.")
#     print(df_concat)

# asyncio.run(main())
# Now once you get this graph and save it, makes sure you can get data synchronously and then start the query time tests for that...

# Also you need to examine the quality of the pandas table you're getting to figure out how to implement your comb op alg

# %%

# Ok Lets just check that we can even get data from both tables:

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

# async def query_and_store():
#     query = "SELECT * FROM my_table"
#     table_name = "combined_table"
#     start_time = time.time()
#     df = await fetch_from_databases(db1_config, db2_config, query)
#     df.to_csv(f"{table_name}.csv", index=False)
#     elapsed_time = time.time() - start_time
#     print(f"Elapsed time: {elapsed_time} seconds")

# loop = asyncio.get_event_loop()
# loop.run_until_complete(query_and_store())
# %%
