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

import networkx as nx
import datetime
from networkx.readwrite import json_graph
import hashlib
from collections import OrderedDict
import collections
import sqlalchemy
from sqlalchemy import create_engine
from tqdm import tqdm
import json
import pandas as pd
import numpy
import re
from collections import defaultdict
from IPython.display import clear_output

import logging
import inspect
import linecache

logPathFileName = './logs/' + '03.log'

# create logger'
logger = logging.getLogger('03')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{logPathFileName}", mode='w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def log_next_line(lines=1, level=logger.info):
    '''
    TODO: this will always log again this same function when called

    :param lines: how many relative lines ahead to log (negative for previous)
    :param level: logger function with log level to log
    :return: None, but logs stuff
    '''
    for i in range(1, lines+1):
        level(linecache.getline(
            __file__, inspect.getlineno(inspect.currentframe()) + i))


# This is for jupyter notebook
# from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()
# This is for terminal
tqdm.pandas(desc="Progress")

# # Params

engine = create_engine('postgresql+psycopg2://mtg:mtg@localhost:5432/mtg')
logger.info(linecache.getline(
    __file__, inspect.getlineno(inspect.currentframe()) + 1))
engine.connect()

# # Helping functions

# + code_folding=[0]
# Split dataframelist


def splitDataFrameList(df, target_column, separator=None):
    '''
    https://gist.github.com/jlln/338b4b0b55bd6984f883
    df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
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
    df.apply(splitListToRows, axis=1, args=(
        new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


# + code_folding=[0]
# Create hashable dict


class HashableDict(OrderedDict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def hexdigext(self):
        return hashlib.sha256(''.join([str(k)+str(v) for k, v in self.items()]).encode()).hexdigest()


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
    return HashableDict({'entity': key}).hexdigext()


# + code_folding=[0]
# function to draw a graph to png
shapes = ['box', 'polygon', 'ellipse', 'oval',
          'circle', 'egg', 'triangle', 'exagon', 'star']
colors = ['blue', 'black', 'red', '#db8625',
          'green', 'gray', 'cyan', '#ed125b']
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
        entity_node_ent_type = attrs.get('entity_node_ent_type', numpy.nan)
        if not pd.isnull(entity_node_ent_type):
            color = entities_colors[entity_node_ent_type.strip('"')]
            node.set_fillcolor(color)
            node.set_color(color)
            node.set_shape('hexagon')
            # node.set_colorscheme()
            node.set_style('filled')

        node_type = attrs.get('type', None)
        if node_type == '"card"':
            color = '#999966'
            node.set_fillcolor(color)
#             node.set_color(color)
            node.set_shape('star')
            # node.set_colorscheme()
            node.set_style('filled')
    #
        # pass

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


# ### Get paths from cards_text to entities (simple paths from text -> entities)
# + deletable=false editable=false run_control={"frozen": true}
with engine.connect() as con:
    try:
        con.execute('''DROP TABLE public.cards_graphs_as_json ''')
    except Exception as e:
        pass
    con.execute('''CREATE TABLE public.cards_graphs_as_json AS
                   (SELECT * FROM public.cards_graphs_as_json_temp)''')

# +
table_name = 'cards_graphs_as_json'
to_table_name = 'cards_text_to_entity_simple_paths'
chunk_size = 200

all_ids = pd.read_sql_query('SELECT card_id from {0}'.
                            format(table_name),
                            engine,
                            )
chunks = [all_ids.iloc[all_ids.index[i:i + chunk_size]]
          for i in range(0, all_ids.shape[0], chunk_size)]


# + code_folding=[]
def get_df_for_subgraphs_of_paths_from_card_to_entities(row):
    '''INPUT: a row with outgoing graph of card
    RETURNs: a dataframe with each row corresponding to a path from text to entity'''
    if not row['outgoing']:
        return []
    G = json_graph.node_link_graph(json.loads(row['outgoing']))
    card_nodes = [x for x, y in G.nodes(data=True) if y['type'] == 'card']
    entity_nodes = [x for x, y in G.nodes(data=True) if y['type'] == 'entity']
    assert len(card_nodes) == 1

    new_rows = []
    for entity_node in entity_nodes:
        paths_list = nx.all_simple_paths(G, card_nodes[0], entity_node)
        for path in paths_list:
            graph_row = {}
            path_g = G.subgraph(path)

            graph_row['card_id'] = row['card_id']
            graph_row['path_graph_json'] = json.dumps(
                json_graph.node_link_data(path_g))
            graph_row['part'] = G.nodes[path[1]]['part']
            has_add = re.findall(r'add ', str(
                graph_row['part']), flags=re.IGNORECASE)
            graph_row['has_add'] = True if has_add else False

            # Text type and orders
    #         graph_row['paragraph_type'] = G.node[path[1]]['paragraph_type']
            graph_row['paragraph_order'] = G.nodes[path[1]]['paragraph_order']
            graph_row['pop_type'] = G.nodes[path[1]]['pop_type']
            graph_row['pop_order'] = G.nodes[path[1]]['pop_order']
            graph_row['part_type'] = G.nodes[path[1]]['part_type']
            graph_row['part_order'] = G.nodes[path[1]]['part_order']

            graph_row['path_text_key'] = (graph_row['card_id']
                                          + '-' +
                                          str(int(
                                              graph_row['paragraph_order']))
                                          + '-' +
                                          str(int(graph_row['pop_order']))
                                          + '-' +
                                          str(int(graph_row['part_order']))
                                          )

            # Entities info
            graph_row['entity_node_entity'] = G.nodes[path[-1]
                                                      ]['entity_node_entity']
            graph_row['entity_node_ent_type'] = G.nodes[path[-1]
                                                        ]['entity_node_ent_type']
            graph_row['entity_node_desc'] = G.nodes[path[-1]
                                                    ]['entity_node_desc']
            graph_row['entity_node_lemma'] = G.nodes[path[-1]
                                                     ]['entity_node_lemma']

            graph_row['path_pk'] = (graph_row['path_text_key']
                                    + '-'+graph_row['entity_node_entity']
                                    )

            graph_row['entity_pos'] = G.nodes[path[-2]]['token_node_pos']
            graph_row['entity_tag'] = G.nodes[path[-2]]['token_node_tag']
            graph_row['entity_head_dep'] = G.nodes[path[-2]]['token_head_dep']

            # Entities head info
            if G.nodes[path[-3]]['type'] == 'token':
                graph_row['entity_head'] = G.nodes[path[-3]]['token_node_text']
                graph_row['entity_head_tag'] = G.nodes[path[-3]
                                                       ]['token_node_tag']
                graph_row['entity_head_head_dep'] = G.nodes[path[-2]
                                                            ]['token_head_dep']
                graph_row['entity_head_pos'] = G.nodes[path[-2]
                                                       ]['token_node_pos']

            # Append row
            new_rows.append(graph_row)

    return pd.DataFrame(new_rows)


# + code_folding=[0] deletable=false editable=false run_control={"frozen": true}
# # Testing dataframe composition
# row = df.iloc[1]
# G = json_graph.node_link_graph(json.loads(row['outgoing']))
# card_nodes = [x for x,y in G.nodes(data=True) if y['type']=='card']
# entity_nodes = [x for x,y in G.nodes(data=True) if y['type']=='entity']
# assert len(card_nodes) == 1
#
# new_rows = []
# for entity_node in entity_nodes:
#     paths_list = nx.all_simple_paths(G, card_nodes[0], entity_node)
#     for path in paths_list:
#         graph_row = {}
#         path_g = G.subgraph(path)
#
#         graph_row['card_id'] = row['card_id']
#         graph_row['path_graph_json'] = json.dumps(json_graph.node_link_data(path_g))
#         graph_row['part'] = G.node[path[1]]['part']
#
#         # Text type and orders
# #         graph_row['paragraph_type'] = G.node[path[1]]['paragraph_type']
#         graph_row['paragraph_order'] = G.node[path[1]]['paragraph_order']
#         graph_row['pop_type'] = G.node[path[1]]['pop_type']
#         graph_row['pop_order'] = G.node[path[1]]['pop_order']
#         graph_row['part_type'] = G.node[path[1]]['part_type']
#         graph_row['part_order'] = G.node[path[1]]['part_order']
#
#         # Entities info
#         graph_row['entity_node_entity'] = G.node[path[-1]]['entity_node_entity']
#         graph_row['entity_node_ent_type'] = G.node[path[-1]]['entity_node_ent_type']
#         graph_row['entity_node_desc'] = G.node[path[-1]]['entity_node_desc']
#
#         graph_row['entity_pos'] = G.node[path[-2]]['token_node_pos']
#         graph_row['entity_tag'] = G.node[path[-2]]['token_node_tag']
#         graph_row['entity_head_dep'] = G.node[path[-2]]['token_head_dep']
#
#         # Entities head info
#         if G.node[path[-3]]['type'] == 'token':
#             graph_row['entity_head'] = G.node[path[-3]]['token_node_text']
#             graph_row['entity_head_tag'] = G.node[path[-3]]['token_node_tag']
#             graph_row['entity_head_head_dep'] = G.node[path[-2]]['token_head_dep']
#             graph_row['entity_head_pos'] = G.node[path[-2]]['token_node_pos']
#
#
#
#         # Append row
#         new_rows.append(graph_row)
# n = pd.DataFrame(new_rows)
# -
logger.info(linecache.getline(
    __file__, inspect.getlineno(inspect.currentframe()) + 1))
df = pd.read_sql_query('SELECT * from {0} WHERE card_id IN ({1})'.
                       format(table_name, ','.join(
                           ["'"+x+"'" for x in chunks[0]['card_id']])),
                       engine,
                       index_col='card_id')


# + code_folding=[]
# Iter chunks and save simple paths
start = datetime.datetime.now()
logger.info(linecache.getline(
    __file__, inspect.getlineno(inspect.currentframe()) + 1))
for i, chunk in enumerate(chunks):

    logger.info(linecache.getline(
        __file__, inspect.getlineno(inspect.currentframe()) + 1))
    df = pd.read_sql_query('SELECT * from {0} WHERE card_id IN ({1})'.
                           format(table_name, ','.join(
                               ["'"+x+"'" for x in chunk['card_id']])),
                           engine,
                           )

    logger.info(linecache.getline(
        __file__, inspect.getlineno(inspect.currentframe()) + 1))
    paths_series = df.progress_apply(
        get_df_for_subgraphs_of_paths_from_card_to_entities, axis='columns')

    # Drop these ids to append them again
#     DROP_QUERY = ('DELETE FROM {0} WHERE card_id IN ({1})'.
#                            format(table_name, ','.join(["'"+x+"'" for x in df.index]))
#                  )
#     print(engine.execute(DROP_QUERY))

    # Create columns if not exists
#     NEW_QUERY = '''
#         ALTER TABLE {0} ADD COLUMN {1} json;
#             '''.format(table_name, 'list_of_subgraphs_of_paths_from_card_to_entities')
#     try:
#         print(engine.execute(NEW_QUERY))
#     except sqlalchemy.exc.ProgrammingError:
#         # Just ignore, it alredy exists
#         pass

    logger.info('Concatenating')
    df = (pd.concat(paths_series.values, sort=False)
          .reset_index(drop=True)
          .set_index(['card_id', 'paragraph_order', 'pop_order', 'part_order', 'entity_node_entity'])
          )

    method = 'append' if i else 'replace'
    df.to_sql(to_table_name, engine, if_exists=method,
              dtype={'path_graph_json': sqlalchemy.types.JSON})

    logger.info('Chunk {0}/{1} ELAPSED: {2}'.format(i,
                len(chunks), datetime.datetime.now()-start))
    logger.info('Export finished')
    if not i % 15:
        clear_output()


# + code_folding=[0] deletable=false editable=false hideCode=false run_control={"frozen": true}
# # Testing
# card_id = some_ids.iloc[0]
# G = json_graph.node_link_graph(json.loads(df.loc[card_id, 'outgoing']))
# card_nodes = [x for x,y in G.nodes(data=True) if y['type']=='card']
# entity_nodes = [x for x,y in G.nodes(data=True) if y['type']=='entity']
# assert len(card_nodes) == 1
#
# paths = []
# for entity_node in entity_nodes:
#     paths_list = nx.all_simple_paths(G, card_nodes[0], entity_node)
#     a = [json_graph.node_link_data(G.subgraph(path)) for path in paths_list]
#     paths.extend(a)
#
# json.dumps(paths)

# + deletable=false editable=false run_control={"frozen": true}
# test = pd.read_sql_query('SELECT * from {0}'.
#                        format(to_table_name),
#                        engine,
#                       index_col=['card_id', 'paragraph_order', 'pop_order', 'part_order', 'entity_node_entity'])
# test
# -

logger.info(f'FINISHED: {__file__}')
