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
# 1. Determine an indirectly measurable power measure
# 1.1 Suggestions: scrape decks and check which cards are most commonly seen
# 2. Engineer features for cards
# 2.1 P/T, cost, supertype, type, subtype, cmc, colored cmc
# 2.2 Token based: verbs, parts types, effects, number of tokens
# 3. Build model to predict measured power (let's assume that power goes from -1 to 1)
#    and get SHAP values/ feature importance to understand which features are most important
# 4. Engineer feature for every card in magic and predict it's power
# 5. Inspect the most powerful ones to confirm it detects power
#
# NEXT
# 4. Don't know yet
#
# **DESIRED RESULT**:
# Cards with a predicted POWER LEVEL

import sqlalchemy
from sqlalchemy import create_engine
from tqdm import tqdm
import copy
import itertools
import json
import pandas as pd
import re
from collections import defaultdict
from IPython.display import clear_output
import networkx as nx
import re
from networkx.readwrite import json_graph
import json
import datetime

import logging
import inspect
import linecache
import os

try:
    __file__
except NameError:
    # for running in ipython
    fname = "04a_2_card_sinergy.py.py"
    __file__ = os.path.abspath(os.path.realpath(fname))

logPathFileName = "./logs/" + "04a.log"

# create logger'
logger = logging.getLogger("04a")
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

engine = create_engine(config.DB_STR)
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
engine.connect()


logger.info(f"FINISHED: {__file__}")
