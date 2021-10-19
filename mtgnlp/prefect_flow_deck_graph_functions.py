# %% Definitions
#
# # The idea here is to
# (see flow_deck_graph in prefect_flows)
#
# 1. Load the previously ETLelled outgoing and incoming graphs
# 2. Build simple paths from card to its entity nodes
# 3. Build a paths df keyed by card_id, entity and orders with some common attributes of these paths:
# paragraph type/order, pop type/order, part type/order,
# entity pos (actualy head's pos), entity head (actually head's head)
# 4. Store in postgres
# 5. Draw graph

from mtgnlp import config
import networkx as nx
import itertools
from networkx.readwrite import json_graph
import sqlalchemy
from sqlalchemy import create_engine
from tqdm import tqdm
import copy
import itertools
import json
import textwrap

import pandas as pd
import numpy
import re

from typing import List
import logging
import inspect
import linecache
import os

try:
    __file__
except NameError:
    # for running in ipython
    fname = "prefect_flow_deck_graph_functions.py"
    __file__ = os.path.abspath(os.path.realpath(fname))

logPathFileName = "./logs/" + "prefect_flow_deck_graph_functions.log"

# create logger'
logger = logging.getLogger("prefect_flow_deck_graph_functions")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{logPathFileName}", mode="w")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# This is for jupyter notebook
# from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()
# This is for terminal
tqdm.pandas(desc="Progress")

# # Params

logger.info("CREATE ENGINE")
ENGINE = create_engine(config.DB_STR)

CARDS_TNAME = config.CARDS_TNAME
CARDS_JSON_TNAME = config.CARDS_JSON_TNAME
DECKS_TNAME = config.DECKS_TNAME
CARDS_TEXT_TO_ENTITY_SIMPLE_PATHS_TNAME = config.CARDS_TEXT_TO_ENTITY_SIMPLE_PATHS_TNAME
DECKS_GRAPH_TNAME = config.DECKS_GRAPH_TNAME

# # Helping functions

# + code_folding=[0]
# function to draw a graph to png
SHAPES = [
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
COLORS = ["blue", "black", "red", "#db8625", "green", "gray", "cyan", "#ed125b"]
STYLES = ["filled", "rounded", "rounded, filled", "dashed", "dotted, bold"]

ENTITIES_COLORS = {
    "PLAYER": "#FF6E6E",
    "ZONE": "#F5D300",
    "ACTION": "#1ADA00",
    "VERBAL_ACTION": "#1ADA00",
    "MANA": "#00DA84",
    "SUBTYPE": "#0DE5E5",
    "TYPE": "#0513F0",
    "SUPERTYPE": "#8D0BCA",
    "NATURE": "#1ADA",
    "ABILITY": "#cc3300",
    "COLOR": "#666633",
    "STEP": "#E0E0F8",
    "PT": "#C10AC1",
    "OBJECT": "#F5A40C",
}
ABSENT_COLOR = "#a87f32"


def relayout(pygraph):
    """Given a graph (pygraphviz), redesign its layout"""

    for i, node in enumerate(pygraph.nodes()):
        attrs = node.attr
        entity_node_ent_type = attrs.get("entity_node_ent_type", None)
        if (not pd.isnull(entity_node_ent_type)) and entity_node_ent_type:
            color = ENTITIES_COLORS.get(entity_node_ent_type.strip('"'), ABSENT_COLOR)
            node.attr["fillcolor"] = color
            node.attr["color"] = color
            node.attr["shape"] = "hexagon"
            node.attr["style"] = "filled"

        node_type = attrs.get("type", None)
        if node_type == '"card"':
            color = "#999966"
            node.attr["fillcolor"] = color
            node.attr["shape"] = "star"
            node.attr["style"] = "filled"

    return pygraph


def draw_graph(G, filename="test.png"):
    pygv = nx.drawing.nx_agraph.to_agraph(G)  # pygraphviz

    pygv = relayout(pygv)

    pygv.layout(prog="dot")
    pygv.draw(filename, prog="neato")

    # from IPython.display import Image
    # return Image(filename)


# -

# # Build graph with Networkx


def load_cards_as_dataframe(
    cards_slugs: list,
    cards_table_name=CARDS_TNAME,
    cards_json_table_name=CARDS_JSON_TNAME,
    engine=ENGINE,
) -> pd.DataFrame:

    logger.info(f"load_cards_as_dataframe: load cards: {cards_slugs}")

    query = f"""
        SELECT * from {cards_json_table_name} as cjson
        JOIN (SELECT name as card_name, id as card_id, type, name_slug FROM {cards_table_name}) as cards
        ON cjson.card_id = cards.card_id
        WHERE cards.name_slug IN ({", ".join([f"'{x}'" for x in cards_slugs])})
    """
    logger.debug(query)

    df = pd.read_sql_query(query, engine)

    if not df.shape[0]:
        logger.error("Empty result from load_cards_as_dataframe")
        df = pd.read_sql_query(f"SELECT name_slug from {CARDS_TNAME} LIMIT 5", engine)
        raise Exception(
            f"""
            The resulting query was empty. Are you sure these cards slugs exist?
            Valid names slugs examples: {df['name_slug'].values}
        """
        )

    df

    return df


def load_decks_cards_as_dataframe(
    deck_id: str,
    cards_table_name=CARDS_TNAME,
    cards_json_table_name=CARDS_JSON_TNAME,
    decks_tname=DECKS_TNAME,
    main_deck="MAIN",
    engine=ENGINE,
) -> pd.DataFrame:

    logger.info(f"load_decks_cards_as_dataframe: load cards from deck: {deck_id}")

    query = f"""
        SELECT * from {cards_json_table_name} as cjson
        JOIN (SELECT name as card_name_2, id as card_id_2, type, text, power, toughness
              FROM {cards_table_name}) as cards
        ON cjson.card_id = cards.card_id_2
        JOIN {decks_tname} as decks
        ON cards.card_name_2 = decks.card_name
        WHERE decks.deck_id = '{deck_id}' AND decks.in = '{main_deck}'
    """
    logger.debug(query)

    df = pd.read_sql_query(query, engine)

    if not df.shape[0]:
        logger.error("Empty result from load_decks_cards_as_dataframe")
        query = f"""
            SELECT DISTINCT deck_id from {decks_tname} as decks
        """
        decks_ids_df = pd.read_sql_query(query, engine)
        raise Exception(
            f"""
            The resulting query was empty. Are you sure this deck_id is registered?
            Add the deck txt to decks folder and run decks ETL no NLP to register it.
            Valid deck ids: {decks_ids_df['deck_id'].values}
        """
        )

    # logger.info(
    #     f"Filter out Basic Lands for now, the graph is too big with them")
    # df = df[~df['type'].str.contains('Basic Land')]
    df

    return df


def aggregate_cards_and_set_weights(
    df: pd.DataFrame,
    # the unique identifier in the input dataframe
    card_id_col="card_id_in_deck",
    # the identifier of the card in the resulting dataframe
    index_name="card_unit_id",
) -> pd.DataFrame:
    """Reduce number of cards to unique cards, setting weight (column) equal
    to the number of times the card appears in the deck/df"""

    counter = df[["card_id", card_id_col]]
    counter = (
        counter.groupby("card_id")
        .count()
        .reset_index()
        .rename(columns={card_id_col: "weight"})
    )
    df2 = df.drop_duplicates(subset=["card_id"]).merge(counter, on=["card_id"])

    if df2.shape[0] > 40:
        raise ValueError(
            f"Too many unique cards in deck ({df2.shape[0]}). "
            f"I can only process up to 40 at this moment."
        )

    df2[index_name] = range(df2.shape[0])

    return df2


def create_cards_graphs(
    # df: one row per card (pk col name: card_id_in_deck), with columns incoming, outgoing
    df: pd.DataFrame,
) -> pd.DataFrame:  # must contain cols {card_id_in_deck, incoming_graph, outgoing_graph}

    df["incoming_graph"] = df["incoming"].progress_apply(
        lambda x: json_graph.node_link_graph(json.loads(x))
    )

    df["outgoing_graph"] = df["outgoing"].progress_apply(
        lambda x: json_graph.node_link_graph(json.loads(x))
    )

    return df


def collapse_single_path(digraph, path):
    """

    :param digraph: networkx.DiGraph
    :param path: list of nodes (simple path of digraph)
    :return: networkx.DiGraph with only first and last nodes and one edge between them

    The original graph is an attribute of the edge
    """
    digraph_ordered = digraph.subgraph(
        path
    )  # in each element node 0 is card, node 1 is text part
    res = nx.DiGraph()
    # Add first and last nodes with their respective attributes
    res.add_node(path[0], **digraph.nodes[path[0]])
    res.add_node(path[-1], **digraph.nodes[path[-1]])
    # edge_attr = {'full_original_path_graph': digraph}
    edge_attr = {}
    labels = []
    short_labels = []

    for i, node in enumerate(path):
        label = ""
        short_label = ""
        if not i:
            continue
        # dict: attributes of each edge in order
        e_at = digraph_ordered.edges[path[i - 1], node]
        edge_attr[f"edge-{i}"] = e_at
        label += e_at.get("part_type_full", None) or e_at.get("label") + ":"
        if dict(digraph_ordered[node]):
            # dict: attributes of each node in order
            n_at = dict(digraph_ordered.nodes[node])
            edge_attr[f"node-{i}"] = dict(digraph_ordered.nodes[node])
            label += n_at.get("label")
            if (e_at.get("type", None) == "token_to_head_part") and (
                e_at.get("label", None) == "ROOT"
            ):
                short_label += f"{n_at.get('token_node_text')} in {n_at.get('part_type') or ''} of {n_at.get('pop_type') or ''}"

        labels.append(label)
        if short_label:
            short_labels.append(short_label)

    res.add_edge(
        path[0],
        path[-1],
        **edge_attr,
        # This label is too big to show when plotting a full deck
        title=("".join(textwrap.wrap(f'{" |<br>".join(labels)}'))),
        label=("".join(textwrap.wrap(f'{" | ".join(short_labels)}'))),
    )

    return res


# %% Compose all


def compose_all_graphs_collapsed(
    # must cointain cols { {node_id_col}, incoming_graph, outgoing_graph}
    df: pd.DataFrame,
    # the column which contains the card node id
    node_id_col: str = "card_id_in_deck",
    weight_col: str = "weight",
    target="card",  # card or entity
) -> nx.Graph:
    """Build all simple paths between cards in the deck, or cards an entities,
    and collpase each simple path into an edge between cards or card and entity"""

    # If there is not weight definition for nodes, set it as 1
    if weight_col not in df.columns:
        df[weight_col] = 1

    # reset incoming and outgoing graphs for the card nodes to have the
    # same id and that id set to card_id_in_deck
    # select node of type==card, get its id, create mapping={id: card_id_in_deck}
    df["incoming_card_original_label"] = df["incoming_graph"].progress_apply(
        lambda g: [n for n, d in g.nodes(data=True) if d["type"] == "card"][0]
    )
    df["outgoing_card_original_label"] = df["outgoing_graph"].progress_apply(
        lambda g: [n for n, d in g.nodes(data=True) if d["type"] == "card"][0]
    )
    # Relabel card nodes to their id in the deck
    df["incoming_graph"] = df.progress_apply(
        lambda row: nx.relabel_nodes(
            row["incoming_graph"],
            {row["incoming_card_original_label"]: row[node_id_col]},
        ),
        axis="columns",
    )

    df["outgoing_graph"] = df.progress_apply(
        lambda row: nx.relabel_nodes(
            row["outgoing_graph"],
            {row["outgoing_card_original_label"]: row[node_id_col]},
        ),
        axis="columns",
    )

    # For pyvis layout: set
    # group=card_type,
    # title=hover_text(whatever I want),
    # size=weight,
    # label=some short name ({weight} card_name)
    def get_label(row):
        if row["power"]:
            return (
                f"{row['weight']} {row['card_name']} {row['power']}/{row['toughness']} "
            )
        return f"{row['weight']} {row['card_name']}"

    for graph_col in ["incoming_graph", "outgoing_graph"]:
        nothing_returned = df.progress_apply(
            lambda row: nx.set_node_attributes(
                row[graph_col],
                # {node_id: {attr_name: value}}
                {
                    row[node_id_col]: {
                        "group": row["type"],
                        "title": row["text"],
                        "size": row["weight"],
                        "weight": row["weight"],
                        "label": get_label(row),
                        # To show card image on hover
                        # 'title': '''<img src="https://c1.scryfall.com/file/scryfall-cards/normal/front/b/f/bf87803b-e7c6-4122-add4-72e596167b7e.jpg" width="150">''',
                    }
                },
            ),
            axis="columns",
        )

    # Compose graph with all incoming and outgoing graphs
    # TODO this does not work, because all simple paths will include paths that don't actually exist
    # for example, it generates this: https://drive.google.com/file/d/1mmpore-FLxWZwxQ0TDZjeTyvLpTA8Mnb/view?usp=sharing
    # in which worhsip points to aura of silence
    # we should instead build all simple paths between every pair of cards or a card and individual entities
    # than compose all simple paths
    def get_all_target_nodes(target_type: str = "card", df=df):
        if target_type == "card":
            return list(df[node_id_col].unique())
        elif target_type == "entity":
            array_of_entity_nodes = df["outgoing_graph"].progress_apply(
                lambda g: [n for n, d in g.nodes(data=True) if d["type"] == "entity"]
            )
            return set([x for lis in array_of_entity_nodes for x in lis])

    def get_targets(
        node_id, get_all_target_nodes=get_all_target_nodes, target_type=target
    ):
        """Return list of target nodes for simple paths"""
        return [x for x in get_all_target_nodes(target_type) if x != node_id]

    def get_all_collapsed_simple_paths(
        node_id, df=df, get_targets=get_targets, target_type=target
    ):
        """Return a list of graphs containing only two nodes:
        node_id and target_node_id,
        with edges representing simple paths between them
        """
        two_node_graphs_list = []
        # for each target_node_id
        for target_node_id in get_targets(node_id):
            #   compose graph G1 from outgoing node_id to incoming target_node_id
            if target_type == "card":
                out = df.loc[df[node_id_col] == node_id, "outgoing_graph"].iloc[0]
                incom = df.loc[
                    df[node_id_col] == target_node_id, "incoming_graph"
                ].iloc[0]
                G1 = nx.algorithms.operators.compose_all([out, incom])
            if target_type == "entity":
                G1 = df.loc[df[node_id_col] == node_id, "outgoing_graph"].iloc[0]
            #   get all simple paths in G1 from node_id to target_node_id
            simpaths = nx.all_simple_paths(G1, node_id, target_node_id)
            #   collapse all simple paths in G1 to generate G2 (only two nodes, multiple edges)
            collapsed_spaths = [collapse_single_path(G1, path) for path in simpaths]
            if collapsed_spaths:
                G2 = nx.algorithms.operators.compose_all(collapsed_spaths)
                #   extend collapsed_paths_list
                two_node_graphs_list.append(G2)
        return two_node_graphs_list

    # Get a column with all two nodes graphs in it
    logger.info(f"Get a column with all two nodes graphs in it")
    df["two_node_graphs_list"] = df[node_id_col].progress_apply(
        get_all_collapsed_simple_paths
    )

    all_two_node_graphs_list = [
        g for graph_list in df["two_node_graphs_list"].values for g in graph_list
    ]

    logger.info(f"Compose all_two_node_graphs_list")
    H = nx.algorithms.operators.compose_all(all_two_node_graphs_list)

    return H


def save_decks_graphs_to_db(
    deck_ids: List[str],  # deck_slug in database
    engine=ENGINE,
    decks_tname=DECKS_TNAME,
    decks_graphs_tname=DECKS_GRAPH_TNAME,
    target="card",
) -> List[nx.Graph]:
    """Calculate graphs for deck_ids
    and save them (serialized do json) to db associated with the deck_id.
    Also, return a list of the generated graphs
    """

    logger.info(f"Saving decks graphs: {deck_ids}")
    res = []
    graphs = []
    for deck_id in deck_ids:
        logger.info(f"Saving deck graph: {deck_id}")
        df_orig = load_decks_cards_as_dataframe(deck_id)
        df = create_cards_graphs(df_orig.copy())
        df = aggregate_cards_and_set_weights(df)
        H = compose_all_graphs_collapsed(df, node_id_col="card_unit_id", target=target)
        graphs.append(H)

        res = pd.DataFrame(
            [
                {
                    "deck_id": deck_id,
                    f"graph_json": json.dumps(json_graph.node_link_data(H)),
                    "graph_target": target,
                }
            ]
        )
        res = res.set_index("deck_id")

        # Perform upsert
        try:
            res.to_sql(decks_graphs_tname, engine, if_exists="fail")
        except ValueError:

            # def create_col_query(col):
            #     return f"""
            #             ALTER TABLE {decks_graphs_tname}
            #             ADD COLUMN IF NOT EXISTS {col} JSON;
            #             """

            delete_query = f"""
                DELETE from {decks_graphs_tname}
                WHERE deck_id = '{deck_id}' AND graph_target = '{target}'
            """
            # WHERE deck_id IN ({", ".join([f"'{x}'" for x in deck_ids])})
            with engine.connect() as con:
                con.execute(delete_query)

                # for col in res.columns:
                #     if col not in [deck_id]:
                #         con.execute(create_col_query(col))

            res.to_sql(decks_graphs_tname, engine, if_exists="append")

        logger.info(f"Finished deck: {deck_id}")

    return graphs


def load_decks_graphs_from_db(
    deck_ids: List[str],  # deck_slug in database
    engine=ENGINE,
    decks_graphs_tname=DECKS_GRAPH_TNAME,
    target="card",
) -> List[nx.Graph]:
    """Load decks graphs as json and de-serialize them to nx.Graphs"""

    res = []
    graphs = []
    query = f"""
    SELECT *
    FROM {decks_graphs_tname}
    WHERE deck_id IN ({", ".join([f"'{x}'" for x in deck_ids])})
    AND   graph_target = '{target}'
    """
    df = pd.read_sql_query(query, engine)
    df["graph"] = df[f"graph_json"].apply(
        lambda x: json_graph.node_link_graph(json.loads(x))
    )

    return list(df["graph"].values)


# %% Draw deck graph
if False:
    # deckids = ['00deck_frustrado_dano_as_is']
    deckids = [
        # '00deck_frustrado_dano_as_is',
        # '00deck_passarinhos_as_is',
        "00deck_alsios_combado"
    ]
    target = "entity"
    H = save_decks_graphs_to_db(deck_ids=deckids, target=target)[0]
    G = load_decks_graphs_from_db(deck_ids=deckids, target=target)[0]
    assert nx.is_isomorphic(H, G)
    draw_graph(G, config.PICS_DECKS_GRAPHS_DIR.join(f"{deckids[0]}_{target}.png"))


# %% Draw two cards graph (it will draw left to right)
if False:
    # cards_slugs = ['incinerate', 'pardic_firecat']
    # cards_slugs = ['aura_of_silence', 'worship']
    # cards_slugs = ['thunderbolt', 'pardic_firecat']
    # cards_slugs = ['swords_to_plowshares', 'white_knight']
    cards_slugs = ["worship", "aura_of_silence"]
    df_orig = load_cards_as_dataframe(cards_slugs)
    df = create_cards_graphs(df_orig.copy())
    outgoing = df.loc[df["name_slug"] == cards_slugs[0], "outgoing_graph"].values[0]
    incoming = df.loc[df["name_slug"] == cards_slugs[1], "incoming_graph"].values[0]
    G = nx.algorithms.operators.compose_all([outgoing, incoming])
    draw_graph(G, config.PICS_DIR.join(f'2_cards/{"-".join(cards_slugs)}.png'))

# %%
