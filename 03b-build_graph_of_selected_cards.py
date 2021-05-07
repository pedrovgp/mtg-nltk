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

# # Build graph for each path from a cards text to entities

# The idea here is to
#
# 1. Load the previously ETLelled outgoing and incoming graphs
# 2. Build simple paths from card to its entity nodes
# 3. Build a paths df keyed by card_id, entity and orders with some common attributes of these paths:
# paragraph type/order, pop type/order, part type/order,
# entity pos (actualy head's pos), entity head (actually head's head)
# 4. Store in postgres
#
# NEXT
# 4. Don't know yet
#
# **DESIRED RESULT**:
# result = dataframe/postgres table: (Indexes: card_id, orders, entity)
#
# | card_id | paragraph_order | pop_order | part_order | entity | paragraph_type | pop_type | part_type | entity_pos | entity_head | main_verb_of_path |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
# | a2fh34 | 0 | 1 | 1 | TYPE: Instant | activated | effect | intensifier | pobj | for | destroy |
import copy
import itertools
import json
import textwrap

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from IPython.display import clear_output

import logging
import inspect
import linecache
import os

try:
    __file__
except NameError:
    # for running in ipython
    fname = '03b-build_graph_of_selected_cards.py'
    __file__ = os.path.abspath(os.path.realpath(fname))

logPathFileName = './logs/' + '03b.log'

# create logger'
logger = logging.getLogger('03b')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{logPathFileName}", mode='w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# This is for jupyter notebook
# from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()
# This is for terminal
from tqdm import tqdm
tqdm.pandas(desc="Progress")

# # Params

from sqlalchemy import create_engine
import sqlalchemy
engine = create_engine('postgresql+psycopg2://mtg:mtg@localhost:5432/mtg')
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
engine.connect()

# # Helping functions

# + code_folding=[0]
# function to draw a graph to png
shapes = ['box', 'polygon', 'ellipse', 'oval', 'circle', 'egg', 'triangle', 'exagon', 'star']
colors = ['blue', 'black', 'red', '#db8625', 'green', 'gray', 'cyan', '#ed125b']
styles = ['filled', 'rounded', 'rounded, filled', 'dashed', 'dotted, bold']

entities_colors = {
    'PLAYER': '#FF6E6E',
    'ZONE': '#F5D300',
    'ACTION': '#1ADA00',
    'VERBAL_ACTION': '#1ADA00',
    'MANA': '#00DA84',
    'SUBTYPE': '#0DE5E5',
    'TYPE': '#0513F0',
    'SUPERTYPE': '#8D0BCA',
    'NATURE': '#1ADA'  ,
    'ABILITY': '#cc3300',
    'COLOR': '#666633',
    'STEP': '#E0E0F8',
    'PT': '#C10AC1',
    'OBJECT': '#F5A40C',
}
ABSENT_COLOR = '#a87f32'

def relayout(pygraph):
    """Given a graph (pygraphviz), redesign its layout"""

    for i, node in enumerate(pygraph.nodes()):
        attrs = node.attr
        entity_node_ent_type = attrs.get('entity_node_ent_type', None)
        if (not pd.isnull(entity_node_ent_type)) and entity_node_ent_type:
            color = entities_colors.get(entity_node_ent_type.strip('"'), ABSENT_COLOR)
            node.attr['fillcolor'] = color
            node.attr['color'] = color
            node.attr['shape'] = 'hexagon'
            node.attr['style'] = 'filled'

        node_type = attrs.get('type', None)
        if node_type == '"card"':
            color = '#999966'
            node.attr['fillcolor'] = color
            node.attr['shape'] = 'star'
            node.attr['style'] = 'filled'

    return pygraph

def draw_graph(G, filename='test.png'):
    pygv = nx.drawing.nx_agraph.to_agraph(G)  # pygraphviz

    pygv = relayout(pygv)

    pygv.layout(prog='dot')
    pygv.draw(filename)

    # from IPython.display import Image
    # return Image(filename)
# -

# # Build graph with Networkx

import networkx as nx
import re

from networkx.readwrite import json_graph
import json
import datetime

# ### Get paths from cards_text to entities (simple paths from text -> entities)

# +
table_name = 'cards_graphs_as_json'
to_table_name = 'cards_text_to_entity_simple_paths'

CARD_LIMIT = 100
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
df = pd.read_sql_query(f'''SELECT * from {table_name} ORDER BY random() LIMIT {CARD_LIMIT}''',
# WHERE card_id IN ({','.join(["'"+x+"'" for x in chunks[0]['card_id']])})''',
                           engine,
                           index_col='card_id')

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
df2 = pd.read_sql_query(f'''
SELECT * from {table_name} as a
JOIN cards as cards
ON cards.id = a.card_id
WHERE cards.name IN ('Terror', 'Soltari Visionary')
LIMIT 10''',
   engine,
   index_col='card_id')

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
df_tempest = pd.read_sql_query(f'''
SELECT * from {table_name} as a
JOIN cards as cards
ON cards.id = a.card_id
WHERE cards.set IN ('TMP')''',
   engine,
   index_col='card_id')
logger.info(f'df_tempest shape: {df_tempest.shape}')

# +
# # Graph building

# +
# # One card
card_id = df.index[0]
G = json_graph.node_link_graph(json.loads(df.loc[card_id, 'outgoing']))
draw_graph(G, 'pics/01-card.png')

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
# # One card - 100 random cards
# for idx in range(CARD_LIMIT):
#     cid = df.index[idx]
#     temp = json_graph.node_link_graph(json.loads(df.loc[cid, 'outgoing']))
#     card_node_attr = [y for x, y in temp.nodes(data=True) if y['type'] == 'card']
#     name = card_node_attr[0]['card_name']
#     draw_graph(temp, f'pics/1_card/{name}.png')

# +
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
# # One card Tempest
def draw_graphs_from_df(row):
    temp = json_graph.node_link_graph(json.loads(row['outgoing']))
    card_node_attr = [y for x, y in temp.nodes(data=True) if y['type'] == 'card']
    name = card_node_attr[0]['card_name']
    draw_graph(temp, f'pics/tempest/{name}.png')

df_tempest.apply(draw_graphs_from_df, axis=1)

# # Two cards
card1_id, card2_id = df2.index[0], df2.index[1]
g1out = json_graph.node_link_graph(json.loads(df2.loc[card1_id, 'outgoing']))
g1in = json_graph.node_link_graph(json.loads(df2.loc[card1_id, 'incoming']))
draw_graph(g1out, 'pics/03a-card1out.png')
g2out = json_graph.node_link_graph(json.loads(df2.loc[card2_id, 'outgoing']))
g2in = json_graph.node_link_graph(json.loads(df2.loc[card2_id, 'incoming']))
draw_graph(g2in, 'pics/03a-card2in.png')

card_nodes = [x for x,y in G.nodes(data=True) if y['type']=='card']
entity_nodes = [x for x,y in G.nodes(data=True) if y['type']=='entity']
assert len(card_nodes) == 1

# # One card simple paths with cleaned text
# Normalize part nodes
# The first nodes (part nodes)
# should be replaced by only a node describing part and pop types
# to simplify comparison between cards
# Attr on this node should be stored in the edge
card_nodes_ids = [x for x, y in G.nodes(data=True) if y['type'] == 'card']  # should be one
part_nodes_ids = [x for x, y in G.nodes(data=True) if y['type'] == 'part']  #  => 0
logger.info(card_nodes)
assert len(card_nodes_ids) == 1
card_node_id = card_nodes_ids[0]
logger.info(part_nodes_ids)
new_graph = nx.DiGraph(G)
for part_node_id in part_nodes_ids:
    new_graph.remove_edge(card_node_id, part_node_id)
    old_attrs = copy.deepcopy(new_graph.nodes[part_node_id])
    logger.info(old_attrs)
    del old_attrs['label']
    new_graph.add_edge(card_node_id, part_node_id, **old_attrs)
    new_id = old_attrs['pop_type'] + '-' + old_attrs['part_type']
    new_graph = nx.relabel.relabel_nodes(new_graph, mapping={part_node_id: new_id})
    attrs = {new_id: {'pop_type': old_attrs['pop_type'],
                      'part_type': old_attrs['part_type'],
                      'label': new_id}}
    nx.set_node_attributes(new_graph, attrs)

H = new_graph

draw_graph(H, 'pics/01-effects_paths.png')


# # One card simple paths
paths = []
for entity_node in entity_nodes:
    paths_list = nx.all_simple_paths(G, card_nodes[0], entity_node)
    a = [G.subgraph(path) for path in paths_list]
    paths.extend(a)

H = nx.algorithms.operators.compose_all(paths)

draw_graph(H, 'pics/02-simple-paths.png')


# # One card simple paths with cleaned text
# Normalize part nodes
# For every path between card and entity, the first node
# should be replaced by only a node describing part and pop types
# to simplify comparison between cards
# Attr on this node should be stored in the edge
paths = []
for entity_node in entity_nodes:
    paths_list = nx.all_simple_paths(G, card_nodes[0], entity_node)
    a = [G.subgraph(path) for path in paths_list]
    a_ordered = [nx.topological_sort(g) for g in a]  # in each element node 0 is card, node 1 is text part
    nodes_0_ids = [next(itertools.islice(g, 0, None)) for g in a_ordered]
    nodes_1_ids = [next(itertools.islice(g, 0, None)) for g in a_ordered]
    b = []
    for i, graph in enumerate(a):
        logger.debug(graph.nodes[nodes_0_ids[i]])
        logger.debug(graph.nodes[nodes_1_ids[i]])
        new_graph = nx.DiGraph(graph)
        new_graph.remove_edge(nodes_0_ids[i], nodes_1_ids[i])
        old_attrs = copy.deepcopy(new_graph.nodes[nodes_1_ids[i]])
        del old_attrs['label']
        logger.info(old_attrs)
        new_graph.add_edge(nodes_0_ids[i], nodes_1_ids[i], **old_attrs)
        new_id = old_attrs['pop_type'] + '-' + old_attrs['part_type']
        new_graph = nx.relabel.relabel_nodes(new_graph, mapping={nodes_1_ids[i]: new_id})
        attrs = {new_id: {'pop_type': old_attrs['pop_type'],
                          'part_type': old_attrs['part_type'],
                          'label': new_id}}
        nx.set_node_attributes(new_graph, attrs)
        b.append((new_graph))
    paths.extend(b)

H = nx.algorithms.operators.compose_all(paths)

draw_graph(H, 'pics/02-normalized_simple_paths.png')

# # Two cards, two nodes, edge with all info
# For every path between card1 and card2, all intermediary nodes
# should be replaced by only an edge encoding all (meaninfull) info
def collapse_single_path(digraph, path):
    '''

    :param digraph: networkx.DiGraph
    :param path: list of nodes (simple path of digraph)
    :return: networkx.DiGraph with only first and last nodes and one edge between them

    The original graph is an attribute of the edge
    '''
    digraph_ordered = digraph.subgraph(path)  # in each element node 0 is card, node 1 is text part
    res = nx.DiGraph()
    # Add first and last nodes with their respective attributes
    res.add_node(path[0], **digraph.nodes[path[0]])
    res.add_node(path[-1], **digraph.nodes[path[-1]])
    # edge_attr = {'full_original_path_graph': digraph}
    edge_attr = {}
    labels = []

    for i, node in enumerate(path):
        label = ''
        if not i:
            continue
        e_at = digraph_ordered.edges[path[i - 1], node]  # dict: attributes of each edge in order
        edge_attr[f'edge-{i}'] = e_at
        label += e_at.get('part_type_full', None) or e_at.get('label') + ':'
        if dict(digraph_ordered[node]):
            n_at = dict(digraph_ordered.nodes[node])   # dict: attributes of each node in order
            edge_attr[f'node-{i}'] = dict(digraph_ordered.nodes[node])
            label += n_at.get('label')

        labels.append(label)

    res.add_edge(path[0], path[-1], **edge_attr, label=''.join(textwrap.wrap(f'{" | ".join(labels)}')))

    return res


G = nx.algorithms.operators.compose_all([g1out, g2in])
draw_graph(G, 'pics/03a-g1out-g2in.png')
paths_list = list(nx.all_simple_paths(G, card1_id, card2_id))
H = nx.algorithms.operators.compose_all([collapse_single_path(G, path) for path in paths_list])

draw_graph(H, 'pics/03-2-cards-2-nodes.png')

logger.info(f'FINISHED: {__file__}')
