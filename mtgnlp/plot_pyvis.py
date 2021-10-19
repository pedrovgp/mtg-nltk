# %% Importing
from mtgnlp import config
from pyvis.network import Network
import networkx as nx
from prefect_flow_deck_graph_functions import (
    save_decks_graphs_to_db,
    load_decks_graphs_from_db,
    draw_graph,
    ENGINE,
)

# %% Options for visual layout


def get_options():

    return """
    var options = {
  "nodes": {
    "color": {
      "hover": {
        "border": "rgba(31,94,181,1)",
        "background": "rgba(169,184,206,1)"
      }
    },
    "scaling": {
      "min": 9
    },
    "shadow": {
      "enabled": true
    }
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 0.2
      }
    },
    "color": {
      "inherit": true
    },
    "smooth": false
  },
  "interaction": {
    "hover": true,
    "multiselect": true
  },
  "manipulation": {
    "enabled": true
  },
  "physics": {
    "enabled": false,
    "forceAtlas2Based": {
      "centralGravity": 0.02,
      "springLength": 100,
      "damping": 0.47,
      "avoidOverlap": 0.04
    },
    "minVelocity": 0.76,
    "solver": "forceAtlas2Based"
  }
}
    """

    return """
var options = {
  "nodes": {
    "borderWidth": 2,
    "borderWidthSelected": 5,
    "color": {
      "highlight": {
        "background": "rgba(210,229,255,0.47)"
      },
      "hover": {
        "border": "rgba(77,86,163,0.75)"
      }
    },
    "shadow": {
      "enabled": true
    },
    "shapeProperties": {
      "borderRadius": 4
    },
    "size": 32
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true
      }
    },
    "color": {
      "inherit": true
    },
    "smooth": {
      "forceDirection": "none"
    }
  },
  "interaction": {
    "hover": true
  },
  "physics": {
    "minVelocity": 0.75
  }
}
    """


# %% Create pyvis Network
nt = Network("800px", "700px")  # , notebook=True)
# nt.set_options(get_options())
# nt.show_buttons(filter_=['physics'])
nt.show_buttons()

# %% Actual deck graph
deck_ids = [
    # '00deck_frustrado_dano_as_is',
    # '00deck_passarinhos_as_is',
    # '00deck_alsios_combado',
    "pv_white",
]
# ds = pd.read_sql('decks', ENGINE, columns=['deck_id'])
# deck_ids = list(ds.deck_id.unique())
target = "entity"
# save_decks_graphs_to_db(deck_ids=deck_ids, target=target)
for dcid in deck_ids:
    G = load_decks_graphs_from_db(deck_ids=[dcid], target=target)[0]
    G.remove_nodes_from(
        [
            n
            for n, d in G.nodes(data=True)
            if d.get("card_name", None) in ["Island", "Plains", "Forbidding Watchtower"]
        ]
    )
    # entity_nodes = [n for n, d in G.nodes(
    #     data=True) if d['type'] == 'entity']
    # nx.set_node_attributes(
    #     G,
    #     # {node_id: {attr_name: value}}
    #     {node_id: {
    #         'x': 100,
    #     } for node_id in entity_nodes
    #     })
    nt.from_nx(G)
    nt.show(config.GRAPHS_DIR.joinpath(f"{dcid}-{target}.html"))
    # nx.set_edge_attributes(
    #     G,
    #     'pays_for',
    #     name='label'
    # )
    draw_graph(G, config.GRAPHS_DIR.joinpath(f"{dcid}-{target}.png"))
