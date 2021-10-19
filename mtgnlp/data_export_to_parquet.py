# -*- coding: utf-8 -*-
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

# # Export postgres tables to parquet (or any other format, actually

# The idea here is to
#
# 1. Read all tables from mtg database into dataframes
# 2. Export dataframes to parquet files
from mtgnlp import config
from sqlalchemy import inspect
from sqlalchemy import create_engine
import json
import pandas as pd
import numpy as np
import re
from collections import defaultdict

import logging
import inspect
import linecache
import os

try:
    __file__
except NameError:
    # for running in ipython
    fname = "01.01-cards-ETL-no-NLP.py"
    __file__ = os.path.abspath(os.path.realpath(fname))

logPathFileName = "./logs/" + "01.01.log"

# create logger'
logger = logging.getLogger("01.01")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{logPathFileName}", mode="w")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# This is for jupyter notebook
# from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()
# This is for terminal
logger.info("Logging to get line number")


engine = create_engine(config.DB_STR)
engine.connect()

logger.info(engine.connect())

inspector = inspect(engine)
schemas = inspector.get_schema_names()

for table_name in inspector.get_table_names(schema="public"):
    logger.info("table_name: %s" % table_name)
    df = pd.read_sql_table(table_name, engine)
    logger.debug(df.head(3))
    df.to_parquet(f"./data/{table_name}.parquet", compression="GZIP")

logger.info(f"FINISHED: {__file__}")
