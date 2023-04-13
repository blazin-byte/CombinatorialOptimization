
# Imports
import PriceSimFxns as psim
import mysql.connector
import pandas as pd
import asyncio
import time
import matplotlib.pyplot as plt
import itertools
import networkx as nx

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

# TEST Synchronous Query Time as Num_Options Increase:


def sync_qtime_num_ops():
    num_options = []
    qtime_sync_options = []
    for i in range(1, 201):
        flights, hotels = psim.generate_time_series(
            num_signals=100*i, signal_length=7)
        num_options.append(3*(i*100))
        psim.upload_to_databases(flights=flights, hotels=hotels)
        qtime_sync_options.append(psim.fetch_data()[1])

    plt.plot(num_options, qtime_sync_options)
    plt.grid()
    plt.xlabel('Number of Options')
    plt.ylabel('Synchronous Query Time')
    plt.show()

# TEST Synchronous Query Time as Days Increase:


def sync_qtime_num_days():
    num_days = []
    qtime_sync_days = []
    for i in range(1, 201):
        flights, hotels = psim.generate_time_series(
            num_signals=100, signal_length=30*i)
        num_days.append(30*i)
        psim.upload_to_databases(flights=flights, hotels=hotels)
        qtime_sync_days.append(psim.fetch_data())

    plt.plot(num_days, qtime_sync_days)
    plt.grid()
    plt.xlabel('Number of Options')
    plt.ylabel('Synchronous Query Time')
    plt.show()


def async_qtime_num_ops():
    ...


def async_qtime_num_days():
    ...


def generate_hotel_vals(values, num_stay):
    all_combinations = itertools.product(values, repeat=num_stay)
    return list(all_combinations)


def bruteforce_1D(df, num_stay):
    start_time = time.time()
    num_ops = df.shape[0]//3
    min_cost = float('inf')
    choices = [0]*(3+num_stay)
    for col_index, col in enumerate(df.columns[:-num_stay]):
        # Passes Column name for column to use
        for fo in range(num_ops):
            hotels_list = generate_hotel_vals(
                range(num_ops, 2*num_ops), num_stay)
            for h_choices in hotels_list:
                for fr in range(2*num_ops, 3*num_ops):
                    outF_p = df[col][fo]
                    hotels_p = 0
                    for i in range(len(h_choices)):
                        hotels_p += df.iloc[h_choices[i], col_index+i]
                    retF_p = df.iloc[fr, col_index+num_stay]
                    curr_cost = outF_p+hotels_p+retF_p
                    if curr_cost < min_cost:
                        min_cost = curr_cost
                        h_print = [df.iloc[[i]].index[0] for i in h_choices]
                        choices = [col, df.iloc[[fo]].index[0]] + \
                            h_print + [df.iloc[[fr]].index[0]]
    end_time = time.time()
    query_time = end_time - start_time
    return (min_cost, choices, query_time)


def proposedAlg1D(df, num_stay):
    start_time = time.time()
    minDf = pd.DataFrame()
    for col in df.columns:
        colAdd = []
        for i in range(0, df.shape[0], 3):
            colAdd.append((df[col][i:i+3].idxmin(), min(df[col][i:i+3])))
        minDf[col] = colAdd
    # Ok now just go over each departure date with trip shape:
    choices = []
    min_cost = float('inf')
    for col_index, col in enumerate(df.columns[:-num_stay]):
        outF_p = minDf[col][0][1]
        hotels_p = 0
        hotel_choices = []
        for i in range(num_stay):
            hotels_p += minDf.iloc[1, col_index+i][1]
            hotel_choices.append(minDf.iloc[1, col_index+i][0])
        retF_p = minDf.iloc[2, col_index+num_stay][1]
        curr_cost = outF_p+hotels_p+retF_p
        if curr_cost < min_cost:
            min_cost = curr_cost
            choices = [col, minDf[col][0][0]] + hotel_choices + \
                [minDf.iloc[2, col_index+num_stay][0]]
    end_time = time.time()
    query_time = end_time - start_time
    return (min_cost, choices, query_time)


def process_df(df, num_options):
    subset = df.iloc[num_options:num_options*2]
    df = df.drop(df.index[num_options:num_options*2])
    df = df.append(subset)
    df = df.set_index(df['type'])
    df = df.drop(df.columns[0], axis=1)
    df = df.astype(float)
    return df


def process_df_2D(df, num_options):
    df = df.set_index(df['type'])
    df = df.drop(df.columns[0], axis=1)
    df = df.astype(float)
    return df


def compare_1D_algs(num_stay):
    num_options_list = []
    brute_force_1D_list = []
    proposed_alg_1D_list = []
    # TODO: For actual tests Eventually set to 201
    for i in range(1, 5):
        flights, hotels = psim.generate_time_series(
            num_signals=100*i, signal_length=7)
        num_options_list.append(3*(i*100))
        psim.upload_to_databases(flights=flights, hotels=hotels)
        df = psim.fetch_data()[0]
        df = process_df(df, num_options=100*i)
        brute_force_1D_list.append(proposedAlg1D(df, num_stay)[2])
        proposed_alg_1D_list.append(bruteforce_1D(df, num_stay)[2])
    return (num_options_list, brute_force_1D_list, proposed_alg_1D_list)
    plt.plot(num_options, qtime_sync_options)
    plt.grid()
    plt.xlabel('Number of Options')
    plt.ylabel('Synchronous Query Time')
    plt.show()


def edge_constructor(input, output, day, minDf, switch=False):
    if (input.endswith('h0')):
        if (output.endswith('h0')):
            return ([], 0)
        elif (output.endswith('d1')):
            cost1 = minDf.iloc[0, day][1] + minDf.iloc[6, day][1]
            cost2 = minDf.iloc[2, day][1] + \
                minDf.iloc[4, day][1] + minDf.iloc[6, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[0, day][0], minDf.iloc[6, day][0]], round(cost1, 2))
            else:
                return ([minDf.iloc[2, day][0], minDf.iloc[4, day][0], minDf.iloc[6, day][0]], round(cost2, 2))
        elif (output.endswith('d2')):
            cost1 = minDf.iloc[2, day][1] + minDf.iloc[7, day][1]
            cost2 = minDf.iloc[0, day][1] + \
                minDf.iloc[3, day][1] + minDf.iloc[7, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[2, day][0], minDf.iloc[7, day][0]], round(cost1, 2))
            else:
                return ([minDf.iloc[0, day][0], minDf.iloc[3, day][0], minDf.iloc[7, day][0]], round(cost2, 2))
    elif (input.endswith('d1')):
        if (output.endswith('d1')):
            if switch:
                return ([], float('inf'))
            return ([minDf.iloc[6, day][0]], round(minDf.iloc[6, day][1], 2))
        elif (output.endswith('d2')):
            cost1 = minDf.iloc[3, day][1] + minDf.iloc[7, day][1]
            cost2 = minDf.iloc[1, day][1] + \
                minDf.iloc[2, day][1] + minDf.iloc[7, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[3, day][0], minDf.iloc[7, day][0]], round(cost1, 2))
            else:
                return ([minDf.iloc[1, day][0], minDf.iloc[2, day][0], minDf.iloc[7, day][0]], round(cost2, 2))
        elif (output.endswith('fin')):
            cost1 = minDf.iloc[1, day][1]
            cost2 = minDf.iloc[3, day][1] + minDf.iloc[5, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[1, day][0]], round(cost1, 2))
            else:
                return ([minDf.iloc[3, day][0], minDf.iloc[5, day][0]], round(cost2, 2))
    elif (input.endswith('d2')):
        if (output.endswith('d1')):
            cost1 = minDf.iloc[4, day][1] + minDf.iloc[6, day][1]
            cost2 = minDf.iloc[5, day][1] + \
                minDf.iloc[0, day][1] + minDf.iloc[6, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[4, day][0], minDf.iloc[6, day][0]], round(cost1, 2))
            else:
                return ([minDf.iloc[5, day][0], minDf.iloc[0, day][0], minDf.iloc[6, day][0]], round(cost2, 2))
        elif (output.endswith('d2')):
            if switch:
                return ([], float('inf'))
            return ([minDf.iloc[7, day][0]], round(minDf.iloc[7, day][1], 2))
        elif (output.endswith('fin')):
            cost1 = minDf.iloc[5, day][1]
            cost2 = minDf.iloc[4, day][1] + minDf.iloc[1, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[5, day][0]], round(cost1, 2))
            else:
                return ([minDf.iloc[4, day][0], minDf.iloc[1, day][0]], round(cost2, 2))


def generate_graph(df, num_days, num_stay, num_options):
    minDf = pd.DataFrame()
    for col in df.columns:
        colAdd = []
        for i in range(0, df.shape[0], num_options):
            colAdd.append((df[col][i:i+num_options].idxmin(),
                           min(df[col][i:i+num_options])))
        minDf[col] = colAdd

    G = nx.DiGraph()

    G.add_node('h0', same=True, stayed=0)
    G.add_node('fin')
    # i will represent the day
    # NOW BUILD IN for add_edge
    for i in range(num_days):
        leaves = [node for node in G.nodes() if G.out_degree(node) == 0]
        for leaf in leaves:
            switch = False
            if leaf.endswith('h0'):
                # If you can stay at home one more day
                if num_days - num_stay > i+1:
                    node_name = leaf+'h0'
                    G.add_node(node_name, same=True, stayed=0)
                    G.add_edge(leaf, node_name, weight=edge_constructor(leaf, node_name, i, minDf)[
                        1], options=edge_constructor(leaf, node_name, i, minDf)[0])
                # Adding d1 and d2 and attributes
                node_name = leaf + 'd1'
                G.add_node(node_name, same=True, stayed=1)
                G.add_edge(leaf, node_name, weight=edge_constructor(leaf, node_name, i, minDf)[
                    1], options=edge_constructor(leaf, node_name, i, minDf)[0])
                node_name = leaf + 'd2'
                G.add_node(node_name, same=True, stayed=1)
                G.add_edge(leaf, node_name, weight=edge_constructor(leaf, node_name, i, minDf)[
                    1], options=edge_constructor(leaf, node_name, i, minDf)[0])

            elif leaf.endswith('d1'):
                if G.nodes[leaf]['stayed'] < num_stay:
                    if (G.nodes[leaf]['stayed'] == num_stay-1 and G.nodes[leaf]['same']):
                        switch = True
                    node_name = leaf + 'd1'
                    G.add_node(node_name, same=True and G.nodes[leaf]['same'],
                               stayed=G.nodes[leaf]['stayed']+1)
                    G.add_edge(leaf, node_name, weight=edge_constructor(leaf, node_name, i, minDf, switch)[
                        1], options=edge_constructor(leaf, node_name, i, minDf, switch)[0])

                    node_name = leaf + 'd2'
                    G.add_node(node_name, same=False and G.nodes[leaf]['same'],
                               stayed=G.nodes[leaf]['stayed']+1)
                    G.add_edge(leaf, node_name, weight=edge_constructor(leaf, node_name, i, minDf)[
                        1], options=edge_constructor(leaf, node_name, i, minDf)[0])
                else:
                    node_name = 'fin'
                    G.add_edge(leaf, node_name, weight=edge_constructor(leaf, node_name, i, minDf)[
                        1], options=edge_constructor(leaf, node_name, i, minDf)[0])

            elif leaf.endswith('d2'):
                if G.nodes[leaf]['stayed'] < num_stay:
                    node_name = leaf + 'd1'
                    G.add_node(node_name, same=False and G.nodes[leaf]['same'],
                               stayed=G.nodes[leaf]['stayed']+1)
                    G.add_edge(leaf, node_name, weight=edge_constructor(leaf, node_name, i, minDf)[
                        1], options=edge_constructor(leaf, node_name, i, minDf)[0])

                    if (G.nodes[leaf]['stayed'] == num_stay-1 and G.nodes[leaf]['same']):
                        switch = True
                    node_name = leaf + 'd2'
                    G.add_node(node_name, same=True and G.nodes[leaf]['same'],
                               stayed=G.nodes[leaf]['stayed']+1)
                    G.add_edge(leaf, node_name, weight=edge_constructor(leaf, node_name, i, minDf, switch)[
                        1], options=edge_constructor(leaf, node_name, i, minDf, switch)[0])
                else:
                    node_name = 'fin'
                    G.add_edge(leaf, node_name, weight=edge_constructor(leaf, node_name, i, minDf)[
                        1], options=edge_constructor(leaf, node_name, i, minDf)[0])
    return G

    # # recursively add nodes to the graph
    # def add_nodes(node, depth):
    #     if depth == num_stay:
    #         # add the leaf node
    #         G.add_edge(node, 'fin')
    #     else:
    #         # add child nodes and recursively call the function
    #         child1 = node + 'a'
    #         child2 = node + 'b'
    #         G.add_edge(node, child1)
    #         G.add_edge(node, child2)
    #         add_nodes(child1, depth+1)
    #         add_nodes(child2, depth+1)

    # # start recursively adding nodes from the starting node
    # add_nodes('a', 1)
    # add_nodes('b', 1)

    # return G

# def create_graph(num_days, num_stay):
#     G = nx.DiGraph()

#     # Create initial node h0
#     G.add_node('h0')


# Now once you fix brute force, you should compare Jianzong to Brute Force...

# num_flights = int(len(df) / 3)
# min_cost = float('inf')
# best_combination = None

# for flight_out_row, flight_out_col, hotel1_row, hotel2_row, return_row, return_col in itertools.product(range(num_flights), range(len(df.columns)), range(num_flights, 2*num_flights), range(num_flights+1, 2*num_flights), range(2*num_flights, 3*num_flights), range(len(df.columns))):
#     if hotel2_row != hotel1_row + 1:
#         continue  # Hotels must be next to each other
#     if return_col != hotel2_row + 1:
#         continue  # Return flight must be in the column after the second hotel

#     total_cost = df.iloc[flight_out_row, flight_out_col] + df.iloc[flight_out_row, flight_out_col+1] + \
#         df.iloc[hotel1_row, flight_out_col+1] + df.iloc[hotel2_row,
#                                                         flight_out_col+1] + df.iloc[return_row, return_col]

#     if total_cost < min_cost:
#         min_cost = total_cost
#         best_combination = (flight_out_row, flight_out_col, hotel1_row,
#                             hotel2_row, return_row, return_col, total_cost)

# if best_combination is not None:
#     out_flight_col = best_combination[1]
#     out_flight_row = best_combination[0]
#     hotel1_row = best_combination[2]
#     hotel2_row = best_combination[3]
#     return_flight_row = best_combination[4]
#     return_flight_col = best_combination[5]
#     total_cost = best_combination[6]

#     return out_flight_row, out_flight_col, hotel1_row, hotel2_row, return_flight_row, return_flight_col, total_cost
# else:
#     return None

# def bruteforce_1D(df):
#     num_flights = int(len(df) / 3)
#     min_cost = float('inf')
#     best_combination = None

#     for flight_out, hotel1, hotel2, flight_return in itertools.product(range(num_flights), repeat=4):
#         if hotel2 != hotel1 + 1:
#             continue  # Hotels must be next to each other
#         if flight_return != hotel2 + 1:
#             continue  # Return flight must be in the column after the second hotel

#         total_cost = df.iloc[flight_out, hotel1] + df.iloc[flight_out, hotel1 +
#                                                            1] + df.iloc[flight_out, hotel2] + df.iloc[flight_return, hotel2+1]

#         if total_cost < min_cost:
#             min_cost = total_cost
#             best_combination = (flight_out, hotel1, hotel2, flight_return)

#     return best_combination, min_cost

# def find_cheapest_combination(df):
#     cheapest_combination = None
#     cheapest_cost = float('inf')

#     for i, row in df.iterrows():
#         date = row['date']
#         outgoing_cost = row[date.day]
#         if pd.isna(outgoing_cost):
#             continue

#         for j in range(i, i+len(df)):
#             hotel_date = df.iloc[j-i]['date']
#             if hotel_date != date:
#                 continue

#             if j > len(df) - 3:
#                 break  # Not enough rows left for return flight and second hotel

#             hotel_columns = [f'price_{col}' for col in df.columns[j:j+2]]
#             if df.loc[j, hotel_columns].isnull().any():
#                 continue

#             for k in range(j+2, j+2+len(df)):
#                 return_date = df.iloc[k-(j+2)]['date']
#                 if return_date != df.iloc[j+1]['date']:
#                     continue

#                 return_column = f'price_{return_date.day}'
#                 if df.loc[k, return_column].isnull():
#                     continue

#                 hotel1_cost = df.loc[j, hotel_columns[0]]
#                 hotel2_cost = df.loc[j+1, hotel_columns[1]]
#                 return_cost = df.loc[k, return_column]
#                 total_cost = outgoing_cost + hotel1_cost + hotel2_cost + return_cost

#                 if total_cost < cheapest_cost:
#                     cheapest_combination = (date, row['flight'],
#                                             df.loc[j, 'hotel'], df.loc[j +
#                                                                        1, 'hotel'],
#                                             df.loc[k, 'flight'], total_cost)
#                     cheapest_cost = total_cost

#     return cheapest_combination
