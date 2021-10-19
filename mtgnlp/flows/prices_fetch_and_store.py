# -*- coding: utf-8 -*-
from mtgnlp import config
from multiprocessing import Pool
import random
import time
import pickle
import slugify
import jsonfinder
import datetime
from lxml import html
from sqlalchemy import inspect
from sqlalchemy import create_engine
import json
import pandas as pd
import numpy as np
import requests
import tqdm

import logging
import inspect
import linecache
import os


logPathFileName = config.LOGS_DIR.joinpath("prices_fetch_and_store.log")

# create logger'
logger = logging.getLogger("prices_fetch_and_store")
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

"""
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
"""
# df.to_sql(tname, engine, if_exists='replace', chunksize=10000, index=True)


tname = "prices"
metatable_name = "mtgprices_cards_urls"
# This will scrape all links again and set status to None (prices will be refetched for them)
REPROCESS_EVERYTHING = False

if REPROCESS_EVERYTHING:

    # 1. Scrape collections urls

    sets_html = requests.get("http://www.mtgprice.com/magic-the-gathering-prices.jsp")
    sets_tree = html.fromstring(sets_html.content.decode())
    sets_elements = sets_tree.xpath("//table[@id='setTable']//a")
    sets_links = ["http://www.mtgprice.com" + a.attrib["href"] for a in sets_elements]

    # 2. For each collection, scrape all cards URLs
    cards_dicts = []
    for set_link in tqdm.tqdm(sets_links):
        card_html = requests.get(set_link)
        card_tree = html.fromstring(card_html.content.decode())
        card_elements = card_tree.xpath("//script[contains(text(), '$scope.setList')]")
        options = list(jsonfinder.jsonfinder(card_elements[0].text))
        cards_list_of_dicts = options[1][2]
        cards_dicts.extend(cards_list_of_dicts)

    with open("cards_links_list.pkl", "wb") as f:
        pickle.dump(cards_dicts, f)

    with open("cards_links_list.pkl", "rb") as f:
        cards_dicts = pickle.load(f)
    logger.info("Loaded cards_dicts from pickle")

    cards_dicts_df = pd.DataFrame(cards_dicts)
    cards_dicts_df["card_url"] = cards_dicts_df["url"].apply(
        lambda x: "http://www.mtgprice.com" + x
    )
    cards_dicts_df["status"] = None
    cards_dicts_df["last_scraped"] = datetime.datetime.now()
    cards_dicts_df = cards_dicts_df.set_index(["card_url"])
    cards_dicts_df.to_sql(metatable_name, engine, if_exists="replace", chunksize=10000)

# 3. For each card URL, apply ETL function


def etl_of_card_prices(card_url, tname="prices", metatable_name=metatable_name):
    # url = 'http://www.mtgprice.com/sets/Zendikar_Expeditions/Misty_Rainforest'
    url = card_url
    logger.info(f"URL: {url}")

    success_reading = False
    while not success_reading:
        try:
            orig_row = pd.read_sql(
                f"""SELECT * FROM public."{metatable_name}" WHERE card_url='{url}'""",
                engine,
            )
            success_reading = True
        except Exception as e:
            time.sleep(random.random() / 20)
            pass
    orig_row = orig_row.set_index(["card_url"])

    collection, card = url.split("/")[-2:]
    t = requests.get(url)
    tree = html.fromstring(t.content.decode())
    prices = tree.xpath("//script[contains(text(), 'var results')]")

    # Process javascript text
    text = prices[0].text.replace("//", "").replace("\n", "").replace("\t", "")
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.replace("}{", "},{").replace("} {", "},{")
    with open("elemt.txt", "w") as f:
        f.write(text)

    texts = [x.strip() for x in text.split("var")]

    # Each stri hold price (results), sellPriceData or VolumeData, in list format, each item of the list is a vendor
    dfs = []
    for stri in texts:
        data_type = stri.split(" ")[0]
        if data_type not in ["results", "sellPriceData", "volumeData"]:
            continue

        eles = []  # this will hold dicts with prices and volume, for each vendor
        for start, end, obj in jsonfinder.jsonfinder(stri):
            if isinstance(obj, dict) and "data" in obj.keys():
                obj["data_type"] = data_type
                obj["card"] = card
                obj["collection"] = collection
                obj["url"] = url
                eles.append(obj)

        # Each dict in eles should be a dictionary of a price source with data holding date, price points
        for ele in eles:
            keys = ele.copy()
            for key in ["color", "lines", "disabled", "data"]:
                if key in keys:
                    keys.pop(key)
            df = pd.DataFrame(ele["data"], columns=["date", "value"])
            for k, v in keys.items():
                df[k] = v
            dfs.append(df.copy())

    dfs = pd.concat(dfs)
    dfs["data_type"] = dfs["data_type"].replace({"results": "price_uss"})
    dfs["date"] = (dfs["date"] / 1000).apply(datetime.datetime.fromtimestamp)
    dfs["vendor"] = dfs["label"].apply(lambda x: x.split("-")[0].strip())
    dfs["card_name_slug"] = dfs["card"].apply(
        lambda x: slugify.slugify(x, separator="_")
    )
    dfs["collection_slug"] = dfs["collection"].apply(
        lambda x: slugify.slugify(x, separator="_")
    )
    dfs = dfs.drop(columns=["label", "name", "collection"], errors="ignore")
    dfs = dfs.set_index(
        ["collection_slug", "card_name_slug", "date", "data_type", "vendor"]
    )

    try:
        dfs.to_sql(tname, engine, if_exists="append", chunksize=10000, index=True)
    except Exception as e:
        orig_row["status"] = f"FAIL: {e}"
        orig_row["last_scraped"] = datetime.datetime.now()
        orig_row.to_sql(metatable_name, engine, if_exists="append")
        logger.info(f"FAIL: {url}")
        logger.info(e)

        return e

    # Throtlle
    logger.info(f"SUCCESS: {url}")
    orig_row["status"] = f"SUCCESS"
    orig_row["last_scraped"] = datetime.datetime.now()
    orig_row.to_sql(metatable_name, engine, if_exists="append")
    time.sleep(random.random() / 10)
    return url


def clean_up():
    #  Drop duplicate from metadata
    # subset = " ".join([f"AND COALESCE(T1.{col}, 'NULL') = COALESCE(T2.{col}, 'NULL')" for col in simple_prices.columns])
    subset = " ".join([f'''AND T1."{col}" = T2."{col}"''' for col in ["card_url"]])
    DROP_DUPLICATES_PG_QUERY = f"""
        DELETE   FROM {metatable_name} T1
          USING       {metatable_name} T2
        WHERE  T1.ctid    < T2.ctid       -- delete the "older" ones
        {subset}  -- list columns that define duplicates
        """
    engine.execute(DROP_DUPLICATES_PG_QUERY)

    # 4. Drop duplicates of prices from prices table, than create views

    #  Drop duplicate prices if we overlapped some dates
    # subset = " ".join([f"AND COALESCE(T1.{col}, 'NULL') = COALESCE(T2.{col}, 'NULL')" for col in simple_prices.columns])
    subset = " ".join(
        [
            f'''AND T1."{col}" = T2."{col}"'''
            for col in [
                "collection_slug",
                "card_name_slug",
                "date",
                "data_type",
                "vendor",
            ]
        ]
    )
    DROP_DUPLICATES_PG_QUERY = f"""
        DELETE   FROM {tname} T1
          USING       {tname} T2
        WHERE  T1.ctid    < T2.ctid       -- delete the "older" ones
        {subset}  -- list columns that define duplicates
        """
    engine.execute(DROP_DUPLICATES_PG_QUERY)


if __name__ == "__main__":
    # for l in tqdm_func(list_to_distribute):
    #     functions.build_graphs_of_cards(l)
    clean_up()

    history = pd.read_sql_table(metatable_name, engine)
    history = history[history["status"] != "SUCCESS"]
    urls_list = list(history["card_url"].unique())
    total = len(urls_list)
    start = datetime.datetime.now()
    with Pool(4) as p:
        logger.info("Starting distributed cards processing")
        # r = list(tqdm.tqdm(p.imap(etl_of_card_prices, urls_list), total=total))
        processed = 0
        for i in p.imap_unordered(etl_of_card_prices, urls_list):
            processed += 1
            logger.info(f"Processed {total}/{processed}: {i}")

    clean_up()

""" # This won't work still, because card names need to be paired (they are probably slugified
engine.execute(f'''DROP VIEW IF EXISTS {tname}_join_cards CASCADE''')
engine.execute(f'''
CREATE VIEW {tname}_join_cards AS
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

engine.execute(f'''DROP VIEW IF EXISTS {tname}_join_cards_ranked CASCADE''')
engine.execute(f'''
 CREATE VIEW {tname}_join_cards_ranked AS
 (SELECT *
  ,DENSE_RANK() over (partition by {tname}_join_cards.name order by {tname}_join_cards.price asc) as rank_cheapest_first
  ,DENSE_RANK() over (partition by {tname}_join_cards.name order by {tname}_join_cards.price desc) as rank_cheapest_last
  , (extract(year from age({tname}_join_cards.date, {tname}_join_cards."set_releaseDate")) * 12
    + extract(month from age({tname}_join_cards.date, {tname}_join_cards."set_releaseDate"))) as months_since_release
  , DATE_PART('day', {tname}_join_cards.date - {tname}_join_cards."set_releaseDate") as days_since_release
 FROM {tname}_join_cards
)
''')
"""
finish = datetime.datetime.now()
logger.info(f"START: {start}")
logger.info(f"FINISHED: {finish}")
logger.info(f"ELAPSED: {finish-start}")
logger.info(f"FINISHED: {__file__}")
