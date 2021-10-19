# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python [conda env:mtg]
#     language: python
#     name: conda-env-mtg-py
# ---

# + deletable=false editable=false run_control={"frozen": true}
# # Build graph for card ids

# + deletable=false editable=false run_control={"frozen": true}
# The idea here is to
#
# 1. Load the previously ETLelled outgoing and incoming nodes and edges (as dataframes)
# 2. Build out and incoming graphs for each card id.
# 3. Maybe build a composed graph (in and out) for each card_id
#
# Next:
# 4. Next, we should work with simple paths from cards to entities (and entities to cards)
#
# At the end, store it in a pickle to avoid parsing everything again next time, which takes a long time.
#
# **DESIRED RESULT**:
# result = {card_id1: {out: graph_from_text, in: graph_from_attributes},
#           card_id2: {out: graph_from_text, in: graph_from_attributes}
#           }
# -

from networkx.readwrite import json_graph
import tqdm
from multiprocessing import Pool
import functions
import networkx as nx
import datetime
import json
import hashlib
from collections import OrderedDict
import collections
import sqlalchemy
from sqlalchemy import create_engine
from tqdm import tqdm
import pandas as pd
import numpy
from collections import defaultdict

import logging
import inspect
import linecache

logPathFileName = "./logs/" + "d_build_individual_cards_graph.log"

# create logger'
logger = logging.getLogger("d_build_individual_cards_graph")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{logPathFileName}", mode="w")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


# This is for jupyter notebook
# from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()
# tqdm_func = tqdm_notebook
# This is for terminal
tqdm.pandas(desc="Progress")
tqdm_func = tqdm

# # Params

engine = create_engine("postgresql+psycopg2://mtg:mtg@localhost:5432/mtg")
logger.info("Logging to get line")
engine.connect()

out_nodes_table_name = "outnodes"
out_edges_table_name = "outedges"
in_nodes_table_name = "innodes"
in_edges_table_name = "inedges"

cards_graphs_as_json_to_table = "cards_graphs_as_json_temp"

# # Create dataframe of cards
logger.info("Logging to get line")
out_nodes = pd.read_sql_table(
    out_nodes_table_name, engine, columns=["card_id", "card_name", "type"]
)
# out_edges = pd.read_sql_table(out_edges_table_name, engine)
# in_nodes = pd.read_sql_table(in_nodes_table_name, engine)
# in_edges = pd.read_sql_table(in_edges_table_name, engine)
# cards_df = cards_df.sample(200)

# +
# ent_out_nodes  = out_nodes[out_nodes['type']=='entity']
# ent_in_nodes = in_nodes[in_nodes['type']=='entity']
# -

# There is no need to build a graph for the same named card twice
unique_cards = out_nodes[out_nodes["type"] == "card"].drop_duplicates(
    subset=["card_name"]
)

ids_to_process = unique_cards["card_id"]  # .sample(11)

# # Helping functions

# + code_folding=[0]
# Split dataframelist


def splitDataFrameList(df, target_column, separator=None):
    """
    https://gist.github.com/jlln/338b4b0b55bd6984f883
    df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    """

    def splitListToRows(row, row_accumulator, target_column, separator):
        split_row = row[target_column]  # .split(separator)
        if isinstance(split_row, collections.Iterable):
            for s in split_row:
                new_row = row.to_dict()
                new_row[target_column] = s
                row_accumulator.append(new_row)
        else:
            new_row = row.to_dict()
            new_row[target_column] = numpy.nan
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(splitListToRows, axis=1, args=(new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


# + code_folding=[0]
# Create hashable dict


class HashableDict(OrderedDict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def hexdigext(self):
        return hashlib.sha256(
            "".join([str(k) + str(v) for k, v in self.items()]).encode()
        ).hexdigest()


# + code_folding=[0]
# Make defaultdict which depends on its key
# Source: https://www.reddit.com/r/Python/comments/27crqg/making_defaultdict_create_defaults_that_are_a/


class key_dependent_dict(defaultdict):
    def __init__(self, f_of_x):
        super().__init__(None)  # base class doesn't get a factory
        self.f_of_x = f_of_x  # save f(x)

    def __missing__(self, key):  # called when a default needed
        ret = self.f_of_x(key)  # calculate default value
        self[key] = ret  # and install it in the dict
        return ret


def entity_key_hash(key):
    return HashableDict({"entity": key}).hexdigext()


# + code_folding=[0]
# function to draw a graph to png
shapes = [
    "box",
    "polygon",
    "ellipse",
    "oval",
    "circle",
    "egg",
    "triangle",
    "exagon",
    "star",
]
colors = ["blue", "black", "red", "#db8625", "green", "gray", "cyan", "#ed125b"]
styles = ["filled", "rounded", "rounded, filled", "dashed", "dotted, bold"]

entities_colors = {
    "PLAYER": "#FF6E6E",
    "ZONE": "#F5D300",
    "ACTION": "#1ADA00",
    "MANA": "#00DA84",
    "SUBTYPE": "#0DE5E5",
    "TYPE": "#0513F0",
    "SUPERTYPE": "#8D0BCA",
    "ABILITY": "#cc3300",
    "COLOR": "#666633",
    "STEP": "#E0E0F8",
}


def draw_graph(G, filename="test.png"):
    pdot = nx.drawing.nx_pydot.to_pydot(G)

    for i, node in enumerate(pdot.get_nodes()):
        attrs = node.get_attributes()
        node.set_label(str(attrs.get("label", "none")))
        #     node.set_fontcolor(colors[random.randrange(len(colors))])
        entity_node_ent_type = attrs.get("entity_node_ent_type", numpy.nan)
        if not pd.isnull(entity_node_ent_type):
            color = entities_colors[entity_node_ent_type.strip('"')]
            node.set_fillcolor(color)
            node.set_color(color)
            node.set_shape("hexagon")
            # node.set_colorscheme()
            node.set_style("filled")

        node_type = attrs.get("type", None)
        if node_type == '"card"':
            color = "#999966"
            node.set_fillcolor(color)
            #             node.set_color(color)
            node.set_shape("star")
            # node.set_colorscheme()
            node.set_style("filled")
    #
    # pass

    for i, edge in enumerate(pdot.get_edges()):
        att = edge.get_attributes()
        att = att.get("label", "NO-LABEL")
        edge.set_label(att)
    #     edge.set_fontcolor(colors[random.randrange(len(colors))])
    #     edge.set_style(styles[random.randrange(len(styles))])
    #     edge.set_color(colors[random.randrange(len(colors))])

    png_path = filename
    pdot.write_png(png_path)

    from IPython.display import Image

    return Image(png_path)


# -

# # Build graph with Networkx


# ## One process (takes 5 hours)

# + deletable=false editable=false run_control={"frozen": true}
# import networkx as nx

# + deletable=false editable=false run_control={"frozen": true}
# def eliminate_and_wrap_in_quotes(text):
#     return '"'+str(text).replace('"', '')+'"'

# + deletable=false editable=false run_control={"frozen": true}
# import pickle

# + code_folding=[0] deletable=false editable=false run_control={"frozen": true}
# # Build out nodes and edges for all ids
# result = {}
# cards_graph_dir = './pickles/card_graphs/'
# import os.path
#
# for i, card_id in enumerate(ids_to_process):
#
#     path_to_graph_file = cards_graph_dir+card_id
#     if os.path.isfile(path_to_graph_file):
#     #if i < 15890:
#         continue
#
#     result[card_id] = {}
#     if not i%100:
#         clear_output()
#     else:
#         if not i%10:
#             print('{0}/{1} cards processed'.format(i, ids_to_process.shape[0]))
#
#     # Card nodes
#     #card_0_nodes = out_nodes[(out_nodes['card_id']==card_id)
#     #                        |(out_nodes['type']=='entity')]
#     card_0_nodes = out_nodes[(out_nodes['card_id']==card_id)]
#     card_0_edges = out_edges[(out_edges['source'].isin(card_0_nodes['node_id']))
#                             |(out_edges['target'].isin(card_0_nodes['node_id']))]
#
#     # Relevant entity nodes
#     ent_0_nodes = ent_out_nodes[ent_out_nodes['node_id'].isin(card_0_edges['source'])
#                                  |ent_out_nodes['node_id'].isin(card_0_edges['target'])]
#
#     card_0_nodes = pd.concat([card_0_nodes, ent_0_nodes], sort=False)
#
#     #result[card_id] = {'nodes': card_0_nodes.copy(), 'edges': card_0_edges.copy()}
#
#     # Build graph
#     edge_attr = [x for x in card_0_edges.columns if not x in ['source', 'target']]
#     G = nx.from_pandas_edgelist(card_0_edges,
#                                 source='source',
#                                 target='target',
#                                 edge_attr=edge_attr,
#                                 create_using=nx.DiGraph())
#
#     ###### IN NODES
#
#     # NODES (set attributes)
#     for k in card_0_nodes['type'].unique():
#         #print(k)
#         #import pdb
#         #pdb.set_trace()
#         node_col = 'node_id'
#         cols = [x for x in card_0_nodes[card_0_nodes['type']==k] if x not in ['node_id']]
#         for node_attr in cols:
#             temp = card_0_nodes[[node_attr, node_col]]
#             temp = temp.dropna()
#
#             # Eliminate and wrap in quotes
#             temp[node_attr] = temp[node_attr].apply(eliminate_and_wrap_in_quotes)
#             nx.set_node_attributes(G, pd.Series(temp[node_attr].values, index=temp[node_col].values).copy().to_dict(), name=node_attr)
#
#     result[card_id]['outgoing'] = G
#
#     # Card nodes
#     card_0_in_nodes = in_nodes[(in_nodes['card_id']==card_id)]
#     card_0_in_edges = in_edges[(in_edges['source'].isin(card_0_in_nodes['node_id']))
#                             |(in_edges['target'].isin(card_0_in_nodes['node_id']))]
#
#     # Relevant entity nodes
#     ent_0_nodes = ent_in_nodes[ent_in_nodes['node_id'].isin(card_0_in_edges['source'])
#                                  |ent_in_nodes['node_id'].isin(card_0_in_edges['target'])]
#
#     card_0_in_nodes = pd.concat([card_0_in_nodes, ent_0_nodes], sort=False)
#
#     #result[card_id] = {'nodes': card_0_in_nodes.copy(), 'edges': card_0_in_edges.copy()}
#
#     # Build graph
#     edge_attr = [x for x in card_0_in_edges.columns if not x in ['source', 'target']]
#     H = nx.from_pandas_edgelist(card_0_in_edges,
#                                 source='source',
#                                 target='target',
#                                 edge_attr=edge_attr,
#                                 create_using=nx.DiGraph())
#
#     # NODES (set attributes)
#     for k in card_0_in_nodes['type'].unique():
#         #print(k)
#         #import pdb
#         #pdb.set_trace()
#         node_col = 'node_id'
#         cols = [x for x in card_0_in_nodes[card_0_in_nodes['type']==k] if x not in ['node_id']]
#         for node_attr in cols:
#             temp = card_0_in_nodes[[node_attr, node_col]]
#             temp = temp.dropna()
#
#             # Eliminate and wrap in quotes
#             temp[node_attr] = temp[node_attr].apply(eliminate_and_wrap_in_quotes)
#             nx.set_node_attributes(H, pd.Series(temp[node_attr].values, index=temp[node_col].values).copy().to_dict(), name=node_attr)
#
#     result[card_id]['incoming'] = H
#
#     a = result[card_id]
#     pickle.dump(a, open(path_to_graph_file, 'wb'))
#     result = {}

# + deletable=false editable=false run_control={"frozen": true}
# draw_graph(result[card_id]['incoming'])

# + deletable=false editable=false hideCode=false run_control={"frozen": true}
# draw_graph(result[card_id]['outgoing'])
# -

# ## JSON in database alternative

# + code_folding=[9, 184] deletable=false editable=false run_control={"frozen": true}
# # Build out nodes and edges for all ids
# result = {}
# cards_graph_dir = './pickles/card_graphs/'
# print_every = 10
# write_every = 200 # cards
#
# start = datetime.datetime.now()
# last=start
#
# for i, card_id in enumerate(ids_to_process):
#
#     result[card_id] = {}
#     if not i%100:
#         clear_output()
#     else:
#         if not i%print_every:
#             print('{0}/{1} cards processed. Elapsed time: {2}'.format(i, ids_to_process.shape[0], datetime.datetime.now()-start))
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Start',now-last))
#     last=now
#
#     # Card nodes
#     card_0_nodes_qr = '''
#     SELECT * FROM {0} WHERE card_id = '{1}'
#     '''.format('public.'+out_nodes_table_name, card_id)
#     card_0_nodes = pd.read_sql_query(card_0_nodes_qr, engine)
#
#     # Card edges
#     card_0_edges_qr = '''
#     SELECT * FROM {0}
#     WHERE source IN {1}
#     OR target IN {1}
#     '''.format('public.'+out_edges_table_name, '('+','.join(["'"+x+"'" for x in card_0_nodes['node_id']])+')')
#     card_0_edges = pd.read_sql_query(card_0_edges_qr, engine)
#
#     # Relevant entity nodes
#     nodes_set = set(numpy.union1d(card_0_edges['source'].values,card_0_edges['target'].values))
#     ent_0_nodes_qr = '''
#     SELECT * FROM {0}
#     WHERE node_id IN {1}
#     AND type='entity'
#     '''.format('public.'+out_nodes_table_name, '('+','.join(["'"+x+"'" for x in nodes_set])+')')
#     ent_0_nodes = pd.read_sql_query(ent_0_nodes_qr, engine)
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Relevant entity nodes filtering',now-last))
#     last=now
#
#     card_0_nodes = pd.concat([card_0_nodes, ent_0_nodes], sort=False)
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Concat',now-last))
#     last=now
#
#     #result[card_id] = {'nodes': card_0_nodes.copy(), 'edges': card_0_edges.copy()}
#
#     # Build graph
#     edge_attr = [x for x in card_0_edges.columns if not x in ['source', 'target']]
#     G = nx.from_pandas_edgelist(card_0_edges,
#                                 source='source',
#                                 target='target',
#                                 edge_attr=edge_attr,
#                                 create_using=nx.DiGraph())
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Build graph',now-last))
#     last=now
#
#     ###### IN NODES
#
#     # NODES (set attributes)
#     for k in card_0_nodes['type'].unique():
#         #print(k)
#         #import pdb
#         #pdb.set_trace()
#         node_col = 'node_id'
#         cols = [x for x in card_0_nodes[card_0_nodes['type']==k] if x not in ['node_id']]
#         for node_attr in cols:
#             temp = card_0_nodes[[node_attr, node_col]]
#             temp = temp.dropna()
#
#             # Eliminate and wrap in quotes
#             #temp[node_attr] = temp[node_attr].apply(eliminate_and_wrap_in_quotes)
#
#             nx.set_node_attributes(G, pd.Series(temp[node_attr].values, index=temp[node_col].values).copy().to_dict(), name=node_attr)
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Set nodes attributes',now-last))
#     last=now
#
#     result[card_id]['outgoing'] = json.dumps(json_graph.node_link_data(G))
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Dump outgoing',now-last))
#     last=now
#
#     # Card in nodes
#     card_0_in_nodes_qr = '''
#     SELECT * FROM {0} WHERE card_id = '{1}'
#     '''.format('public.'+in_nodes_table_name, card_id)
#     card_0_in_nodes = pd.read_sql_query(card_0_in_nodes_qr, engine)
#
#     # Card in edges
#     card_0_in_edges_qr = '''
#     SELECT * FROM {0}
#     WHERE source IN {1}
#     OR target IN {1}
#     '''.format('public.'+in_edges_table_name, '('+','.join(["'"+x+"'" for x in card_0_in_nodes['node_id']])+')')
#     card_0_in_edges = pd.read_sql_query(card_0_in_edges_qr, engine)
#
#     # Relevant entity nodes
#     nodes_set = set(numpy.union1d(card_0_in_edges['source'].values,card_0_in_edges['target'].values))
#     ent_0_nodes_qr = '''
#     SELECT * FROM {0}
#     WHERE node_id IN {1}
#     AND type='entity'
#     '''.format('public.'+in_nodes_table_name, '('+','.join(["'"+x+"'" for x in nodes_set])+')')
#     ent_0_nodes = pd.read_sql_query(ent_0_nodes_qr, engine)
#
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Filter innodes',now-last))
#     last=now
#
#     card_0_in_nodes = pd.concat([card_0_in_nodes, ent_0_nodes], sort=False)
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Concat innodes',now-last))
#     last=now
#
#     #result[card_id] = {'nodes': card_0_in_nodes.copy(), 'edges': card_0_in_edges.copy()}
#
#     # Build graph
#     edge_attr = [x for x in card_0_in_edges.columns if not x in ['source', 'target']]
#     H = nx.from_pandas_edgelist(card_0_in_edges,
#                                 source='source',
#                                 target='target',
#                                 edge_attr=edge_attr,
#                                 create_using=nx.DiGraph())
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Build',now-last))
#     last=now
#
#     # NODES (set attributes)
#     for k in card_0_in_nodes['type'].unique():
#         #print(k)
#         #import pdb
#         #pdb.set_trace()
#         node_col = 'node_id'
#         cols = [x for x in card_0_in_nodes[card_0_in_nodes['type']==k] if x not in ['node_id']]
#         for node_attr in cols:
#             temp = card_0_in_nodes[[node_attr, node_col]]
#             temp = temp.dropna()
#
#             # Eliminate and wrap in quotes
#             #temp[node_attr] = temp[node_attr].apply(eliminate_and_wrap_in_quotes)
#             nx.set_node_attributes(H, pd.Series(temp[node_attr].values, index=temp[node_col].values).copy().to_dict(), name=node_attr)
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Set innodes attr',now-last))
#     last=now
#
#     result[card_id]['incoming'] = json.dumps(json_graph.node_link_data(H))
#
#     now=datetime.datetime.now()
#     print('Took {1}: process {0}'.format('Dump innodes',now-last))
#     last=now
#
#     # Every X cards, create df and write to database
#     X = write_every
#     if (not i%X) and i:
#         print('Exporting to Postgres. Elapsed: {0}'.format(datetime.datetime.now()-start))
#         method = 'append' if (i!=X and i) else 'replace'
#
#         df = pd.DataFrame.from_dict(result, orient='index')
#         df.index.name = 'card_id'
#         df.to_sql(cards_graphs_as_json_to_table, engine, if_exists=method,
#                  dtype = {'outgoing':sqlalchemy.types.JSON, 'incoming':sqlalchemy.types.JSON})
#         result={}
#         print('Exporting to Postgres')
#         print('Exporting to Postgres finished. Elapsed: {0}'.format(datetime.datetime.now()-start))
#
# if result:
#     print('Last exporting to Postgres')
#     df = pd.DataFrame.from_dict(result, orient='index')
#     df.index.name = 'card_id'
#     df.to_sql(cards_graphs_as_json_to_table, engine, if_exists='append',
#               dtype = {'outgoing':sqlalchemy.types.JSON, 'incoming':sqlalchemy.types.JSON})
#     result={}
# print('FINISHED. After: {0}'.format(datetime.datetime.now()-start))
# -

# ## Multiprocessing

# ## First way

# 1. Break the ids list in 4 different ids lists: [[1,2,3],[4,5,6],[7,8,9],[10,11,12]] ([l1,l2,l3,l4])
# 2. Build a list with four tuples: [(l1, table_name, outnodes, outedges, innodes, inedges, engine),
# (l2, table_name, outnodes, outedges, innodes, inedges, engine), (l3, table_name, outnodes, outedges, innodes, inedges, engine), (l4, table_name, outnodes, outedges, innodes, inedges, engine)]. This will pickle the tuple to send to the subprocess, which will copy the objects four times, but that's the way I found out how to do it.
# 3. Process each tuple looping over the ids in l1, l2, l3, l4 (one list in each subprocess)
#
# **IMPORTANT**
# Process the first id independentely to create the table in the database with replace. Than, the processes will only append.

# + code_folding=[8] deletable=false editable=false run_control={"frozen": true}
# # def process_ids(list_with_many_ids,
# #                 cards_graphs_as_json_to_table=cards_graphs_as_json_to_table,
# #                 out_nodes=out_nodes,
# #                 out_edges=out_edges,
# #                 in_nodes=in_nodes,
# #                 in_edges=in_edges,
# #                 engine=engine,
# #                ):
# def process_ids(obj_tuple):
#     list_with_many_ids, cards_graphs_as_json_to_table, out_nodes, out_edges, in_nodes, in_edges,engine = obj_tuple
#     ids_to_process = list_with_many_ids
#
#     result = {}
# #     print_every = 10
#     write_every = 200 # cards
#
#     start = datetime.datetime.now()
#
#     for i, card_id in enumerate(ids_to_process):
#
#         result[card_id] = {}
#         if not i%100:
#             clear_output()
#         else:
#             if not i%print_every:
#                 print('{0}/{1} cards processed. Elapsed time: {2}'.format(i, ids_to_process.shape[0], datetime.datetime.now()-start))
#
#         # Card nodes
#         #card_0_nodes = out_nodes[(out_nodes['card_id']==card_id)
#         #                        |(out_nodes['type']=='entity')]
#         card_0_nodes = out_nodes[(out_nodes['card_id']==card_id)]
#         card_0_edges = out_edges[(out_edges['source'].isin(card_0_nodes['node_id']))
#                                 |(out_edges['target'].isin(card_0_nodes['node_id']))]
#
#         # Relevant entity nodes
#         ent_0_nodes = ent_out_nodes[ent_out_nodes['node_id'].isin(card_0_edges['source'])
#                                      |ent_out_nodes['node_id'].isin(card_0_edges['target'])]
#
#         card_0_nodes = pd.concat([card_0_nodes, ent_0_nodes], sort=False)
#
#         #result[card_id] = {'nodes': card_0_nodes.copy(), 'edges': card_0_edges.copy()}
#
#         # Build graph
#         edge_attr = [x for x in card_0_edges.columns if not x in ['source', 'target']]
#         G = nx.from_pandas_edgelist(card_0_edges,
#                                     source='source',
#                                     target='target',
#                                     edge_attr=edge_attr,
#                                     create_using=nx.DiGraph())
#
#         ###### IN NODES
#
#         # NODES (set attributes)
#         for k in card_0_nodes['type'].unique():
#             #print(k)
#             #import pdb
#             #pdb.set_trace()
#             node_col = 'node_id'
#             cols = [x for x in card_0_nodes[card_0_nodes['type']==k] if x not in ['node_id']]
#             for node_attr in cols:
#                 temp = card_0_nodes[[node_attr, node_col]]
#                 temp = temp.dropna()
#
#                 # Eliminate and wrap in quotes
#                 #temp[node_attr] = temp[node_attr].apply(eliminate_and_wrap_in_quotes)
#
#                 nx.set_node_attributes(G, pd.Series(temp[node_attr].values, index=temp[node_col].values).copy().to_dict(), name=node_attr)
#
#         result[card_id]['outgoing'] = json.dumps(json_graph.node_link_data(G))
#
#         # Card nodes
#         card_0_in_nodes = in_nodes[(in_nodes['card_id']==card_id)]
#         card_0_in_edges = in_edges[(in_edges['source'].isin(card_0_in_nodes['node_id']))
#                                 |(in_edges['target'].isin(card_0_in_nodes['node_id']))]
#
#         # Relevant entity nodes
#         ent_0_nodes = ent_in_nodes[ent_in_nodes['node_id'].isin(card_0_in_edges['source'])
#                                      |ent_in_nodes['node_id'].isin(card_0_in_edges['target'])]
#
#         card_0_in_nodes = pd.concat([card_0_in_nodes, ent_0_nodes], sort=False)
#
#         #result[card_id] = {'nodes': card_0_in_nodes.copy(), 'edges': card_0_in_edges.copy()}
#
#         # Build graph
#         edge_attr = [x for x in card_0_in_edges.columns if not x in ['source', 'target']]
#         H = nx.from_pandas_edgelist(card_0_in_edges,
#                                     source='source',
#                                     target='target',
#                                     edge_attr=edge_attr,
#                                     create_using=nx.DiGraph())
#
#         # NODES (set attributes)
#         for k in card_0_in_nodes['type'].unique():
#             #print(k)
#             #import pdb
#             #pdb.set_trace()
#             node_col = 'node_id'
#             cols = [x for x in card_0_in_nodes[card_0_in_nodes['type']==k] if x not in ['node_id']]
#             for node_attr in cols:
#                 temp = card_0_in_nodes[[node_attr, node_col]]
#                 temp = temp.dropna()
#
#                 # Eliminate and wrap in quotes
#                 #temp[node_attr] = temp[node_attr].apply(eliminate_and_wrap_in_quotes)
#                 nx.set_node_attributes(H, pd.Series(temp[node_attr].values, index=temp[node_col].values).copy().to_dict(), name=node_attr)
#
#         result[card_id]['incoming'] = json.dumps(json_graph.node_link_data(H))
#
#         # Every X cards, create df and write to database
#         X = write_every
#         if (not i%X) and i:
#             print('Exporting to Postgres. Elapsed: {0}'.format(datetime.datetime.now()-start))
#             method = 'append' if (i!=X and i) else 'replace'
#
#             df = pd.DataFrame.from_dict(result, orient='index')
#             df.index.name = 'card_id'
#             df.to_sql(cards_graphs_as_json_to_table, engine, if_exists=method,
#                      dtype = {'outgoing':sqlalchemy.types.JSON, 'incoming':sqlalchemy.types.JSON})
#             result={}
#             print('Exporting to Postgres')
#             print('Exporting to Postgres finished. Elapsed: {0}'.format(datetime.datetime.now()-start))
#
#     if result:
#         print('Last exporting to Postgres')
#         df = pd.DataFrame.from_dict(result, orient='index')
#         df.index.name = 'card_id'
#         df.to_sql(cards_graphs_as_json_to_table, engine, if_exists='append',
#                   dtype = {'outgoing':sqlalchemy.types.JSON, 'incoming':sqlalchemy.types.JSON})
#         result={}
#     print('FINISHED. After: {0}'.format(datetime.datetime.now()-start))
# -

# %load_ext autoreload
# %autoreload 2


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


ids_to_process = list(ids_to_process)
first_tuple = tuple([[ids_to_process[0]]])
list_of_lists_of_ids = list(
    chunks(ids_to_process[1:], int(len(ids_to_process[1:]) / 1000))
)

logger.info("Logging to get line")
functions.build_graphs_of_cards(first_tuple, method="replace")

# +
list_to_distribute = []

for l in list_of_lists_of_ids:
    list_to_distribute.append(tuple([l]))
# list_to_distribute

# +
logger.info("Logging to get line")
if __name__ == "__main__":
    # for l in tqdm_func(list_to_distribute):
    #     functions.build_graphs_of_cards(l)
    total = len(list_to_distribute)
    processed = 0
    with Pool(10) as p:
        # r = list(tqdm_func(p.imap(functions.build_graphs_of_cards, list_to_distribute), total=len(list_to_distribute)))
        # r = list(p.imap(functions.build_graphs_of_cards, list_to_distribute))
        for i in p.imap_unordered(functions.build_graphs_of_cards, list_to_distribute):
            processed += 1
            logger.info(f"Processed {total}/{processed}: {i}")
    # -

    # ### Testing

    # + deletable=false editable=false run_control={"frozen": true}
    # ids_to_process = range(298)

    # + deletable=false editable=false run_control={"frozen": true}
    # def chunks(l, n):
    #     """Yield successive n-sized chunks from l."""
    #     for i in range(0, len(l), n):
    #         yield l[i:i + n]
    #
    # list_of_lists_of_ids = list(chunks(ids_to_process, int(len(ids_to_process)/4)))

    # + deletable=false editable=false run_control={"frozen": true}
    # list_to_distribute = []
    # for l in list_of_lists_of_ids:
    #     print(len(l))
    #     list_to_distribute.append(
    #         tuple(l)#, cards_graphs_as_json_to_table, out_nodes, out_edges, in_nodes, in_edges, engine)
    #     )
    # #list_to_distribute

    logger.info(f"FINISHED: {__file__}")
