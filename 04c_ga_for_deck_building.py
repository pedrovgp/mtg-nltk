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

# # Generate a deck from a given set of cards

# Partly following the idea outlined here:
# https://hackernoon.com/machine-learning-magic-64a6a7f864d4
#
# 1. Determine a system of points for cards and interactions given by deck graph
# 1.1 Suggestions: P/T by CMC, P/T by coloured CMC, number of parts, entities affected,
#     verbs by cost, point for different verbs, etc...
# 1.2 Also: it could be made relative to other cards in the set, for example,
#     How are the cards in the deck affected by text in cards of the whole set?
#     How does these points compare to average points of the rest of the set?
# 2. Define a fitness function as a function of these points
# 2.1 There is room here for human intervention assigning higher weights to
#     whatever she/he wants to favor (speed: lower CMCs, surprises: instants, etc.)
# 3. Use GA algorithms to generate a deck with the given fitness function
#
# NEXT
# 4. Don't know yet

# **DESIRED RESULT**:
# Given a set of cards, a deck list with cards within that set.

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
