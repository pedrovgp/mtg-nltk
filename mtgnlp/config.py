# https://github.com/microsoft/vscode-python/issues/944#issuecomment-773293816
# Create .env file in the root dir of this project
# Append the below to venv/bin/activate and your .env variables will be loaded on activation (opening a new terminal)
# export $(grep -v '^#' .env | xargs)
import os

DB_STR = os.getenv("DB_STR")
DATA_DIR = "data"
CARD_IMAGES_DIR = "card_images"
GRAPHS_DIR = "card_images"
