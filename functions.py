import pandas as pd
import sqlalchemy
import datetime
import networkx as nx
import json
from networkx.readwrite import json_graph

from sqlalchemy import create_engine

write_every = 200 # cards

def build_graphs_of_cards(obj_tuple, method='append'):
    list_with_many_ids, = obj_tuple
    ids_to_process = list_with_many_ids
    print(ids_to_process)
    
    # Params
    engine = create_engine('postgresql+psycopg2://mtg:mtg@localhost:5432/mtg')
    engine.connect()
    
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
        
        result[card_id] = {}

        # Card nodes
        card_0_nodes_qr = '''
        SELECT * FROM {0} WHERE card_id = '{1}'
        '''.format('public.'+out_nodes_table_name, card_id)
        card_0_nodes = pd.read_sql_query(card_0_nodes_qr, engine)

        # Card edges
        card_0_edges_qr = '''
        SELECT * FROM {0}
        WHERE source IN {1}
        OR target IN {1}
        '''.format('public.'+out_edges_table_name, '('+','.join(["'"+x+"'" for x in card_0_nodes['node_id']])+')')
        card_0_edges = pd.read_sql_query(card_0_edges_qr, engine)

        # Relevant entity nodes
        nodes_set = set(pd.np.union1d(card_0_edges['source'].values,card_0_edges['target'].values))
        ent_0_nodes_qr = '''
        SELECT * FROM {0}
        WHERE node_id IN {1}
        AND type='entity'
        '''.format('public.'+out_nodes_table_name, '('+','.join(["'"+x+"'" for x in nodes_set])+')')
        ent_0_nodes = pd.read_sql_query(ent_0_nodes_qr, engine)

        card_0_nodes = pd.concat([card_0_nodes, ent_0_nodes], sort=False)

        #result[card_id] = {'nodes': card_0_nodes.copy(), 'edges': card_0_edges.copy()}

        # Build graph
        edge_attr = [x for x in card_0_edges.columns if not x in ['source', 'target']]
        G = nx.from_pandas_edgelist(card_0_edges,
                                    source='source',
                                    target='target',
                                    edge_attr=edge_attr,
                                    create_using=nx.DiGraph())

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

        result[card_id]['outgoing'] = json.dumps(json_graph.node_link_data(G))

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
        nodes_set = set(pd.np.union1d(card_0_in_edges['source'].values,card_0_in_edges['target'].values))
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
            print('Exporting to Postgres. Elapsed: {0}'.format(datetime.datetime.now()-start))
            method = method

            df = pd.DataFrame.from_dict(result, orient='index')
            df.index.name = 'card_id'
            df.to_sql(cards_graphs_as_json_to_table, engine, if_exists=method,
                     dtype = {'outgoing':sqlalchemy.types.JSON, 'incoming':sqlalchemy.types.JSON})
            result={}
            print('Exporting to Postgres')
            print('Exporting to Postgres finished. Elapsed: {0}'.format(datetime.datetime.now()-start))

    if result:
        print('Last exporting to Postgres')
        df = pd.DataFrame.from_dict(result, orient='index')
        df.index.name = 'card_id'
        df.to_sql(cards_graphs_as_json_to_table, engine, if_exists=method,
                  dtype = {'outgoing':sqlalchemy.types.JSON, 'incoming':sqlalchemy.types.JSON})
        result={}
    print('FINISHED. After: {0}'.format(datetime.datetime.now()-start))