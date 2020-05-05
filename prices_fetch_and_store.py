# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
import requests
import tqdm

import logging
import inspect
import linecache
import os

try:
    __file__
except NameError:
    # for running in ipython
    fname = 'prices_fetch_and_store'
    __file__ = os.path.abspath(os.path.realpath(fname))

logPathFileName = './logs/' + 'prices_fetch_and_store.log'

# create logger'
logger = logging.getLogger('prices_fetch_and_store')
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
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))

from sqlalchemy import create_engine
from sqlalchemy import inspect

engine = create_engine('postgresql+psycopg2://mtg:mtg@localhost:5432/mtg')
engine.connect()

logger.info(engine.connect())

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

url = 'https://www.mtgjson.com/files/AllPrices.json.zip'
save_path = './data/'
fname = 'prices'

download_url(url, save_path+fname+'.zip')
import zipfile
with zipfile.ZipFile(save_path+fname+'.zip', 'r') as zip_ref:
    zip_ref.extractall(save_path)

with open(save_path+'AllPrices.json', 'r') as f:
    prices = json.loads(f.read())

prices_reschemad = []
stop = False
for card_id, pric in tqdm.tqdm(prices.items()):
    # if stop: break
    # print(card_id, pric)
    for useless, medium_dict in pric.items():
        # if stop: break
        # print(useless, medium_dict)
        for medium, date_dict in medium_dict.items():
            # if stop: break
            # print(medium, date_dict)
            # raise Exception('stop')
            if not isinstance(date_dict, dict):  # Dont know why, by uuic sometime appear as a value in medium_dict
                continue
            for date, value in date_dict.items():
                prices_reschemad.append({'date': date, 'card_id': card_id, 'card_media': medium, 'price': value})
                # if stop: break
                # if len(prices_reschemad)> 50: stop = True

df = pd.DataFrame(prices_reschemad)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index(['date', 'card_id', 'card_media'])

logger.info('Saving prices to postgres')
tname = 'prices'
engine.execute(f'''DROP TABLE IF EXISTS {tname} CASCADE''')
df.to_sql(tname, engine, if_exists='replace', chunksize=10000, index=True)

engine.execute(f'''DROP VIEW IF EXISTS {tname}_join_cards CASCADE''')
engine.execute(f'''
CREATE OR REPLACE VIEW {tname}_join_cards AS
     (SELECT *
     FROM {tname} 
     LEFT JOIN (SELECT *
                ,DENSE_RANK() over (partition by cards.name order by cards."set_releaseDate" asc) as rank_oldest_first
                ,DENSE_RANK() over (partition by cards.name order by cards."set_releaseDate" desc) as rank_newest_first
                FROM cards
                ) as cards_enh
     ON prices.card_id=cards_enh.id
    )
''')

engine.execute(f'''
 CREATE OR REPLACE VIEW {tname}_join_cards_ranked AS
 (SELECT *
  ,DENSE_RANK() over (partition by {tname}_join_cards.name order by {tname}_join_cards.price asc) as rank_cheapest_first
  ,DENSE_RANK() over (partition by {tname}_join_cards.name order by {tname}_join_cards.price desc) as rank_cheapest_last
 FROM {tname}_join_cards
)
''')

logger.info(f'FINISHED: {__file__}')
