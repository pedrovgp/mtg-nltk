# https://github.com/microsoft/vscode-python/issues/944#issuecomment-773293816
# Create .env file in the root dir of this project
# Append the below to venv/bin/activate and your .env variables will be loaded on activation (opening a new terminal)
# export $(grep -v '^#' .env | xargs)
import os
from pathlib import Path
import mtgnlp

DB_STR = os.getenv("DB_STR")
ROOT_DIR = Path(mtgnlp.__file__).parent.parent
DATA_DIR = ROOT_DIR.joinpath("data")
CARD_IMAGES_DIR = ROOT_DIR.joinpath("card_images")
PICS_DIR = ROOT_DIR.joinpath("pics")
PICS_DECKS_GRAPHS_DIR = PICS_DIR.joinpath("decks_graphs")
GRAPHS_DIR = ROOT_DIR.joinpath("graphs")
MTGMETA_DECK_TNAME = "mtgmetaio_decks"
MTGMETA_CARDS_TNAME = "mtgmetaio_cards_in_deck"
CARDS_TNAME = "cards"
CARDS_JSON_TNAME = "cards_graphs_as_json"
CARDS_TEXT_TO_ENTITY_SIMPLE_PATHS_TNAME = "cards_text_to_entity_simple_paths"
DECKS_TNAME = "decks"
DECKS_GRAPH_TNAME = "decks_graphs"
DECKS_FEATURES_TNAME = "deck_features"
ALL_PRINTINGS_PATH = DATA_DIR.joinpath("AllPrintings20211017.json")
