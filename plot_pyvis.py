# %% Importing

from pyvis.network import Network
import networkx as nx
from prefect_flow_deck_graph_functions import load_decks_graphs_from_db

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
nt = Network('700px', '700px')  # , notebook=True)
nt.set_options(get_options())
# nt.show_buttons(filter_=['physics'])
# nt.show_buttons()

# %% Example graph
nx_graph = nx.cycle_graph(10)
nx_graph.nodes[1]['title'] = 'Number 1'
nx_graph.nodes[1]['group'] = 1
nx_graph.nodes[3]['title'] = 'I belong to a different group!'
nx_graph.nodes[3]['group'] = 10
nx_graph.add_node(20, size=20, title='couple', group=2)
nx_graph.add_node(21, size=15, title='couple', group=2)
nx_graph.add_edge(20, 21, weight=5)
nx_graph.add_node(25, size=25, label='lonely',
                  title='lonely node', group=3)
# nt.from_nx(nx_graph)
# nt.show(f'./graphs/nx.html')

# %% Actual deck graph
deckids = [
    # '00deck_frustrado_dano_as_is',
    # '00deck_passarinhos_as_is',
    '00deck_alsios_combado'
]
for dcid in deckids:
    G = load_decks_graphs_from_db(deck_ids=[dcid])[0]
    nt.from_nx(G)
    nt.show(f'./graphs/{dcid}.html')
