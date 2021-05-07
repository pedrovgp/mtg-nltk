import pandas as pd
import numpy
import sqlalchemy
import datetime
import networkx as nx
import json
from networkx.readwrite import json_graph

import logging
import inspect
import linecache

logPathFileName = './logs/' + 'functions.log'

# This is for terminal
# from tqdm import tqdm
# tqdm.pandas(desc="Progress")
# tqdm_func = tqdm

from sqlalchemy import create_engine

write_every = 200 # cards

def build_graphs_of_cards(obj_tuple, method='append'):

    # create logger'
    logger = logging.getLogger('functions')
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

    list_with_many_ids, = obj_tuple
    ids_to_process = list_with_many_ids
    f_id = ids_to_process[0]
    logger.debug(f'First id: {f_id}')
    logger.debug(f'{f_id}: {ids_to_process}')

    # Params
    engine = create_engine('postgresql+psycopg2://mtg:mtg@localhost:5432/mtg')
    logger.debug(f'{f_id}: engine creation ran')
    engine.connect()

    logger.debug(f'{f_id}: engine.connect() ran')

    out_nodes_table_name = 'outnodes'
    out_edges_table_name = 'outedges'
    in_nodes_table_name = 'innodes'
    in_edges_table_name = 'inedges'

    cards_graphs_as_json_to_table = 'cards_graphs_as_json_temp'

    # Load
    #out_nodes = pd.read_sql_table(out_nodes_table_name, engine)
    #out_edges = pd.read_sql_table(out_edges_table_name, engine)
    #in_nodes = pd.read_sql_table(in_nodes_table_name, engine)
    #in_edges = pd.read_sql_table(in_edges_table_name, engine)

    #ent_out_nodes  = out_nodes[out_nodes['type']=='entity']
    #ent_in_nodes = in_nodes[in_nodes['type']=='entity']

    result = {}
#     print_every = 10
    write_every = 200 # cards

    start = datetime.datetime.now()

    for i, card_id in enumerate(ids_to_process):

        if not i: logger.debug(f'{f_id}: loop started')

        result[card_id] = {}

        # Card nodes
        card_0_nodes_qr = '''
        SELECT * FROM {0} WHERE card_id = '{1}'
        '''.format('public.'+out_nodes_table_name, card_id)
        card_0_nodes = pd.read_sql_query(card_0_nodes_qr, engine)
        logger.debug(f'{f_id}: Card nodes query ran')

        # Card edges
        card_0_edges_qr = '''
        SELECT * FROM {0}
        WHERE source IN {1}
        OR target IN {1}
        '''.format('public.'+out_edges_table_name, '('+','.join(["'"+x+"'" for x in card_0_nodes['node_id']])+')')
        card_0_edges = pd.read_sql_query(card_0_edges_qr, engine)
        logger.debug(f'{f_id}: Card edges query ran')

        # Relevant entity nodes
        nodes_set = set(numpy.union1d(card_0_edges['source'].values,card_0_edges['target'].values))
        ent_0_nodes_qr = '''
        SELECT * FROM {0}
        WHERE node_id IN {1}
        AND type='entity'
        '''.format('public.'+out_nodes_table_name, '('+','.join(["'"+x+"'" for x in nodes_set])+')')
        ent_0_nodes = pd.read_sql_query(ent_0_nodes_qr, engine)
        logger.debug(f'{f_id}: Relevant entity nodes query ran')

        card_0_nodes = pd.concat([card_0_nodes, ent_0_nodes], sort=False)

        #result[card_id] = {'nodes': card_0_nodes.copy(), 'edges': card_0_edges.copy()}

        # Build graph
        edge_attr = [x for x in card_0_edges.columns if not x in ['source', 'target']]
        G = nx.from_pandas_edgelist(card_0_edges,
                                    source='source',
                                    target='target',
                                    edge_attr=edge_attr,
                                    create_using=nx.DiGraph())

        logger.debug(f'{f_id}: Build graph ran')

        ###### IN NODES

        # NODES (set attributes)
        for k in card_0_nodes['type'].unique():
            #print(k)
            #import pdb
            #pdb.set_trace()
            node_col = 'node_id'
            cols = [x for x in card_0_nodes[card_0_nodes['type']==k] if x not in ['node_id']]
            for node_attr in cols:
                temp = card_0_nodes[[node_attr, node_col]]
                temp = temp.dropna()

                # Eliminate and wrap in quotes
                #temp[node_attr] = temp[node_attr].apply(eliminate_and_wrap_in_quotes)

                nx.set_node_attributes(G, pd.Series(temp[node_attr].values, index=temp[node_col].values).copy().to_dict(), name=node_attr)

        logger.debug(f'''{f_id}: k in card_0_nodes['type'].unique() ran''')
        result[card_id]['outgoing'] = json.dumps(json_graph.node_link_data(G))
        logger.debug(f'''{f_id}: json.dumps ran''')

        # Card in nodes
        card_0_in_nodes_qr = '''
        SELECT * FROM {0} WHERE card_id = '{1}'
        '''.format('public.'+in_nodes_table_name, card_id)
        card_0_in_nodes = pd.read_sql_query(card_0_in_nodes_qr, engine)

        # Card in edges
        card_0_in_edges_qr = '''
        SELECT * FROM {0}
        WHERE source IN {1}
        OR target IN {1}
        '''.format('public.'+in_edges_table_name, '('+','.join(["'"+x+"'" for x in card_0_in_nodes['node_id']])+')')
        card_0_in_edges = pd.read_sql_query(card_0_in_edges_qr, engine)

        # Relevant entity nodes
        nodes_set = set(numpy.union1d(card_0_in_edges['source'].values,card_0_in_edges['target'].values))
        ent_0_nodes_qr = '''
        SELECT * FROM {0}
        WHERE node_id IN {1}
        AND type='entity'
        '''.format('public.'+in_nodes_table_name, '('+','.join(["'"+x+"'" for x in nodes_set])+')')
        ent_0_nodes = pd.read_sql_query(ent_0_nodes_qr, engine)

        card_0_in_nodes = pd.concat([card_0_in_nodes, ent_0_nodes], sort=False)

        #result[card_id] = {'nodes': card_0_in_nodes.copy(), 'edges': card_0_in_edges.copy()}

        # Build graph
        edge_attr = [x for x in card_0_in_edges.columns if not x in ['source', 'target']]
        H = nx.from_pandas_edgelist(card_0_in_edges,
                                    source='source',
                                    target='target',
                                    edge_attr=edge_attr,
                                    create_using=nx.DiGraph())

        # NODES (set attributes)
        for k in card_0_in_nodes['type'].unique():
            #print(k)
            #import pdb
            #pdb.set_trace()
            node_col = 'node_id'
            cols = [x for x in card_0_in_nodes[card_0_in_nodes['type']==k] if x not in ['node_id']]
            for node_attr in cols:
                temp = card_0_in_nodes[[node_attr, node_col]]
                temp = temp.dropna()

                # Eliminate and wrap in quotes
                #temp[node_attr] = temp[node_attr].apply(eliminate_and_wrap_in_quotes)
                nx.set_node_attributes(H, pd.Series(temp[node_attr].values, index=temp[node_col].values).copy().to_dict(), name=node_attr)

        result[card_id]['incoming'] = json.dumps(json_graph.node_link_data(H))

        # Every X cards, create df and write to database
        X = write_every
        if (not i%X) and i:
            logger.debug(f'{f_id}: Exporting to Postgres. Elapsed: {datetime.datetime.now()-start}')
            method = method

            df = pd.DataFrame.from_dict(result, orient='index')
            df.index.name = 'card_id'
            df.to_sql(cards_graphs_as_json_to_table, engine, if_exists=method,
                     dtype = {'outgoing':sqlalchemy.types.JSON, 'incoming':sqlalchemy.types.JSON})
            result={}
            logger.debug(f'{f_id}: Exporting to Postgres finished. Elapsed: {datetime.datetime.now()-start}')

    if result:
        logger.debug(f'{f_id}: Last exporting to Postgres')
        df = pd.DataFrame.from_dict(result, orient='index')
        df.index.name = 'card_id'
        df.to_sql(cards_graphs_as_json_to_table, engine, if_exists=method,
                  dtype = {'outgoing':sqlalchemy.types.JSON, 'incoming':sqlalchemy.types.JSON})
        result={}

    logger.debug(f'{f_id}: FINISHED. After: {datetime.datetime.now()-start}')
    return obj_tuple
