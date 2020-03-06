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

import json
import pandas as pd
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
    'MANA': '#00DA84',
    'SUBTYPE': '#0DE5E5',
    'TYPE': '#0513F0',
    'SUPERTYPE': '#8D0BCA',
    'ABILITY': '#cc3300',
    'COLOR': '#666633',
    'STEP': '#E0E0F8'
}

def draw_graph(G, filename='test.png'):
    pdot = nx.drawing.nx_pydot.to_pydot(G)

    for i, node in enumerate(pdot.get_nodes()):
        attrs = node.get_attributes()
        node.set_label(str(attrs.get('label', 'none')))
    #     node.set_fontcolor(colors[random.randrange(len(colors))])
        entity_node_ent_type = attrs.get('entity_node_ent_type', pd.np.nan)
        if not pd.isnull(entity_node_ent_type):
            color = entities_colors[entity_node_ent_type.strip('"')]
            node.set_fillcolor(color)
            node.set_color(color)
            node.set_shape('hexagon')
            #node.set_colorscheme()
            node.set_style('filled')
        
        node_type = attrs.get('type', None)
        if node_type == '"card"':
            color = '#999966'
            node.set_fillcolor(color)
#             node.set_color(color)
            node.set_shape('star')
            #node.set_colorscheme()
            node.set_style('filled')
    #     
        #pass

    for i, edge in enumerate(pdot.get_edges()):
        att = edge.get_attributes()
        att = att.get('label', 'NO-LABEL')
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

import networkx as nx
import re

from networkx.readwrite import json_graph
import json
import datetime

# ### Get paths from cards_text to entities (simple paths from text -> entities)

# +
table_name = 'cards_graphs_as_json'
to_table_name = 'cards_text_to_entity_simple_paths'
chunk_size = 200

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
df = pd.read_sql_query(f'''SELECT * from {table_name} LIMIT 10''',
# WHERE card_id IN ({','.join(["'"+x+"'" for x in chunks[0]['card_id']])})''',
                           engine,
                           index_col='card_id')

# +
# # Testing
card_id = df.index[0]
G = json_graph.node_link_graph(json.loads(df.loc[card_id, 'outgoing']))
draw_graph(G, 'test.png')

card_nodes = [x for x,y in G.nodes(data=True) if y['type']=='card']
entity_nodes = [x for x,y in G.nodes(data=True) if y['type']=='entity']
assert len(card_nodes) == 1

# paths = []
# for entity_node in entity_nodes:
#     paths_list = nx.all_simple_paths(G, card_nodes[0], entity_node)
#     a = [json_graph.node_link_data(G.subgraph(path)) for path in paths_list]
#     paths.extend(a)
# json.dumps(paths)

paths = []
for entity_node in entity_nodes:
    paths_list = nx.all_simple_paths(G, card_nodes[0], entity_node)
    a = [G.subgraph(path) for path in paths_list]
    paths.extend(a)

import functools
H = functools.reduce(lambda a, b: nx.algorithms.operators.disjoint_union(a, b), paths)

draw_graph(H, 'test2.png')

logger.info(f'FINISHED: {__file__}')
