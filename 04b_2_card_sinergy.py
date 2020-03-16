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
# 1. Determine an indirectly measurable sinergy measure
# 1.1 Suggestions: scrape decks and check which pair of cards is most commonly seen
# 2. Engineer features for pairs of cards
# 2.1 Build card1-entities-card2 simple paths and get graph metrics
# 2.2 Build card2-entities-card1 simple paths and get graph metrics
# 2.3 Compose card1-entities and card2-entities and get graph metrics
# 2.4 Compose entities-card1 and entities-card2 and get graph metrics
# 2.5 Besides graph metrics, counting main verbs or identifying repeating simple paths
#     may be useful somehow
# 3. Build model to predict measured sinergy (let's assume that sinergy goes from -1 to 1)
#    and get SHAP values/ feature importance to understand which features are most important
# 4. Engineer feature for every pair of cards in magic and predict it's sinergy
# 5. Inspect the most sinergical ones to confirm it detects sinergy
#
# NEXT
# 4. Don't know yet
#
# **DESIRED RESULT**:
# Pair of cards with a predicted SINERGY LEVEL

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
    fname = '04a_2_card_sinergy.py.py'
    __file__ = os.path.abspath(os.path.realpath(fname))

logPathFileName = './logs/' + '04a.log'

# create logger'
logger = logging.getLogger('04a')
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

# This is for jupyter notebook
# from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()
# This is for terminal
from tqdm import tqdm
tqdm.pandas(desc="Progress")

# # Params

from sqlalchemy import create_engine
import sqlalchemy
engine = create_engine('postgresql+psycopg2://mtg:mtg@localhost:5432/mtg')
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
engine.connect()




logger.info(f'FINISHED: {__file__}')
