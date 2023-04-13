# %%
import scipy as sp
import numpy as np
import PriceSimFxns as psim
import AlgorithmFxns as algs
import mysql.connector
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import pandas as pd
num_days = 4
num_stay = num_days//2
num_options = 2

flights, hotels = psim.generate_time_series_2D(
    num_signals=num_options, signal_length=num_days)
psim.upload_to_databases(flights=flights, hotels=hotels)
df = psim.fetch_data()[0]
df = algs.process_df_2D(df, num_options=num_options)

# Assume Main Passes This Function df
# %%
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
                G.add_edge(leaf, node_name, weight=algs.edge_constructor(leaf, node_name, i, minDf)[
                           1], options=algs.edge_constructor(leaf, node_name, i, minDf)[0])
            # Adding d1 and d2 and attributes
            node_name = leaf + 'd1'
            G.add_node(node_name, same=True, stayed=1)
            G.add_edge(leaf, node_name, weight=algs.edge_constructor(leaf, node_name, i, minDf)[
                       1], options=algs.edge_constructor(leaf, node_name, i, minDf)[0])
            node_name = leaf + 'd2'
            G.add_node(node_name, same=True, stayed=1)
            G.add_edge(leaf, node_name, weight=algs.edge_constructor(leaf, node_name, i, minDf)[
                       1], options=algs.edge_constructor(leaf, node_name, i, minDf)[0])

        elif leaf.endswith('d1'):
            if G.nodes[leaf]['stayed'] < num_stay:
                if (G.nodes[leaf]['stayed'] == num_stay-1 and G.nodes[leaf]['same']):
                    switch = True
                node_name = leaf + 'd1'
                G.add_node(node_name, same=True and G.nodes[leaf]['same'],
                           stayed=G.nodes[leaf]['stayed']+1)
                G.add_edge(leaf, node_name, weight=algs.edge_constructor(leaf, node_name, i, minDf, switch)[
                           1], options=algs.edge_constructor(leaf, node_name, i, minDf, switch)[0])

                node_name = leaf + 'd2'
                G.add_node(node_name, same=False and G.nodes[leaf]['same'],
                           stayed=G.nodes[leaf]['stayed']+1)
                G.add_edge(leaf, node_name, weight=algs.edge_constructor(leaf, node_name, i, minDf)[
                           1], options=algs.edge_constructor(leaf, node_name, i, minDf)[0])
            else:
                node_name = 'fin'
                G.add_edge(leaf, node_name, weight=algs.edge_constructor(leaf, node_name, i, minDf)[
                           1], options=algs.edge_constructor(leaf, node_name, i, minDf)[0])

        elif leaf.endswith('d2'):
            if G.nodes[leaf]['stayed'] < num_stay:
                node_name = leaf + 'd1'
                G.add_node(node_name, same=False and G.nodes[leaf]['same'],
                           stayed=G.nodes[leaf]['stayed']+1)
                G.add_edge(leaf, node_name, weight=algs.edge_constructor(leaf, node_name, i, minDf)[
                           1], options=algs.edge_constructor(leaf, node_name, i, minDf)[0])

                if (G.nodes[leaf]['stayed'] == num_stay-1 and G.nodes[leaf]['same']):
                    switch = True
                node_name = leaf + 'd2'
                G.add_node(node_name, same=True and G.nodes[leaf]['same'],
                           stayed=G.nodes[leaf]['stayed']+1)
                G.add_edge(leaf, node_name, weight=algs.edge_constructor(leaf, node_name, i, minDf, switch)[
                           1], options=algs.edge_constructor(leaf, node_name, i, minDf, switch)[0])
            else:
                node_name = 'fin'
                G.add_edge(leaf, node_name, weight=algs.edge_constructor(leaf, node_name, i, minDf)[
                           1], options=algs.edge_constructor(leaf, node_name, i, minDf)[0])

        # Weight for last day stay in the same place is inf.
        # Function call for weights
        # Build Dictionary of options
        # Build dictionary inside function connecting selected rows to flights and hotels options


# Plotting
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

path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
    G, source='h0', target='fin')

# print the path
print(path)
# %%

# Ok you got that working...
# Now you need to get choosing min cost and passing flights and hotels back
# Don't forget to use switch
# Maybe make them all return a name and label just so you don't break code later


def edge_constructor(input, output, day, minDf, switch=False):
    if (input.endswith('h0')):
        if (output.endswith('h0')):
            return ([], 0)
        elif (output.endswith('d1')):
            cost1 = minDf.iloc[0, day][1] + minDf.iloc[6, day][1]
            cost2 = minDf.iloc[2, day][1] + \
                minDf.iloc[4, day][1] + minDf.iloc[6, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[0, day][0], minDf.iloc[6, day][0]], cost1)
            else:
                return ([minDf.iloc[2, day][0], minDf.iloc[4, day][0], minDf.iloc[6, day][0]], cost2)
        elif (output.endswith('d2')):
            cost1 = minDf.iloc[2, day][1] + minDf.iloc[7, day][1]
            cost2 = minDf.iloc[0, day][1] + \
                minDf.iloc[3, day][1] + minDf.iloc[7, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[2, day][0], minDf.iloc[7, day][0]], cost1)
            else:
                return ([minDf.iloc[0, day][0], minDf.iloc[3, day][0], minDf.iloc[7, day][0]], cost2)
    elif (input.endswith('d1')):
        if (output.endswith('d1')):
            if switch:
                return ([], float('inf'))
            return ([minDf.iloc[6, day][0]], minDf.iloc[6, day][1])
        elif (output.endswith('d2')):
            cost1 = minDf.iloc[3, day][1] + minDf.iloc[7, day][1]
            cost2 = minDf.iloc[1, day][1] + \
                minDf.iloc[2, day][1] + minDf.iloc[7, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[3, day][0], minDf.iloc[7, day][0]], cost1)
            else:
                return ([minDf.iloc[1, day][0], minDf.iloc[2, day][0], minDf.iloc[7, day][0]], cost2)
        elif (output.endswith('fin')):
            cost1 = minDf.iloc[1, day][1]
            cost2 = minDf.iloc[3, day][1] + minDf.iloc[5, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[1, day][0]], cost1)
            else:
                return ([minDf.iloc[3, day][0], minDf.iloc[5, day][0]], cost2)
    elif (input.endswith('d2')):
        if (output.endswith('d1')):
            cost1 = minDf.iloc[4, day][1] + minDf.iloc[6, day][1]
            cost2 = minDf.iloc[5, day][1] + \
                minDf.iloc[0, day][1] + minDf.iloc[6, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[4, day][0], minDf.iloc[6, day][0]], cost1)
            else:
                return ([minDf.iloc[5, day][0], minDf.iloc[0, day][0], minDf.iloc[6, day][0]], cost2)
        elif (output.endswith('d2')):
            if switch:
                return ([], float('inf'))
            return ([minDf.iloc[7, day][0]], minDf.iloc[7, day][1])
        elif (output.endswith('fin')):
            cost1 = minDf.iloc[5, day][1]
            cost2 = minDf.iloc[4, day][1] + minDf.iloc[1, day][1]
            if cost1 <= cost2:
                return ([minDf.iloc[5, day][0]], cost1)
            else:
                return ([minDf.iloc[4, day][0], minDf.iloc[1, day][0]], cost2)


num_days = 4
num_stay = num_days//2
num_options = 1

flights, hotels = psim.generate_time_series_2D(
    num_signals=num_options, signal_length=num_days)
psim.upload_to_databases(flights=flights, hotels=hotels)
df = psim.fetch_data()[0]
df = algs.process_df_2D(df, num_options=num_options)
minDf = pd.DataFrame()
for col in df.columns:
    colAdd = []
    for i in range(0, df.shape[0], num_options):
        colAdd.append((df[col][i:i+num_options].idxmin(),
                      min(df[col][i:i+num_options])))
    minDf[col] = colAdd
# WOOHOO THIS WORKSSSSS
print(edge_constructor('h0h0', 'h0h0d1', day=1, minDf=minDf, switch=False))

# %%
# Getting min_df working


# Frst lets get generator to do 6*num*signals
# Hotels to be 2* num_signals
num_days = 4
num_stay = num_days//2
num_options = 2

flights, hotels = psim.generate_time_series_2D(
    num_signals=num_options, signal_length=num_days)
psim.upload_to_databases(flights=flights, hotels=hotels)
df = psim.fetch_data()[0]
df = algs.process_df_2D(df, num_options=num_options)
minDf = pd.DataFrame()
for col in df.columns:
    colAdd = []
    for i in range(0, df.shape[0], num_options):
        colAdd.append((df[col][i:i+num_options].idxmin(),
                      min(df[col][i:i+num_options])))
    minDf[col] = colAdd


# %%
in1 = [[0, 1, 0, 0, 0],
       [0, 3, 3, 3, 0],
       [0, 3, 3, 3, 1],
       [1, 3, 2, 3, 0],
       [0, 3, 2, 3, 0],
       [0, 0, 0, 1, 0]]
in2 = [[-1, -1], [1, 1]]
print(sp.signal.correlate2d(in1, in2, mode='valid', boundary='fill', fillvalue=0))

# %%
