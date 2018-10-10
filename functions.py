import pandas as pd
import sqlalchemy
import datetime
import networkx as nx
import json
from networkx.readwrite import json_graph

def process_ids(obj_tuple, method='append'):
    list_with_many_ids, cards_graphs_as_json_to_table, out_nodes, out_edges, in_nodes, in_edges, ent_out_nodes, ent_in_nodes, engine = obj_tuple
    ids_to_process = list_with_many_ids
    
    result = {}
#     print_every = 10
    write_every = 200 # cards

    start = datetime.datetime.now()
    
    for i, card_id in enumerate(ids_to_process):
        
        result[card_id] = {}

        # Card nodes
        #card_0_nodes = out_nodes[(out_nodes['card_id']==card_id)
        #                        |(out_nodes['type']=='entity')]
        card_0_nodes = out_nodes[(out_nodes['card_id']==card_id)]
        card_0_edges = out_edges[(out_edges['source'].isin(card_0_nodes['node_id']))
                                |(out_edges['target'].isin(card_0_nodes['node_id']))]

        # Relevant entity nodes
        ent_0_nodes = ent_out_nodes[ent_out_nodes['node_id'].isin(card_0_edges['source'])
                                     |ent_out_nodes['node_id'].isin(card_0_edges['target'])]

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

        # Card nodes
        card_0_in_nodes = in_nodes[(in_nodes['card_id']==card_id)]
        card_0_in_edges = in_edges[(in_edges['source'].isin(card_0_in_nodes['node_id']))
                                |(in_edges['target'].isin(card_0_in_nodes['node_id']))]

        # Relevant entity nodes
        ent_0_nodes = ent_in_nodes[ent_in_nodes['node_id'].isin(card_0_in_edges['source'])
                                     |ent_in_nodes['node_id'].isin(card_0_in_edges['target'])]

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