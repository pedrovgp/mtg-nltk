from ..helpers.upsert_df import upsert_df
import logging
import os
from typing import List, Set

import pandas as pd
from sqlalchemy import create_engine, engine

from prefect import Flow, Parameter, Task, task

from prefect.tasks.control_flow.filter import FilterTask
from prefect.engine.signals import SKIP


logger = logging.getLogger()

engine = create_engine(os.getenv("DB_STR"))
MTGMETA_DECK_TNAME = "mtgmetaio_decks"
MTGMETA_CARDS_TNAME = "mtgmetaio_cards_in_deck"
DECKS_TNAME = "decks"

# Scrapy crawl flow


class ScrapyCrawlMtgmetaio(Task):
    """Task to crawl mtgmeta.io and save decks and vs_stats
    in tables prefixes by mtgmetaio"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        from datetime import datetime

        os.chdir("./scraper")
        os.system(
            f"scrapy crawl mtgmetaio -O mtgmetaio-{datetime.utcnow():%Y-%m-%d-%H:%M}.json"
        )


# Crawl metgmetaio
flow_scrapy_crawl = Flow("Scrapy-crawl-flow", tasks=[ScrapyCrawlMtgmetaio()])


def get_list_or_all_deck_urls(
    urls: List = [], tname: str = MTGMETA_DECK_TNAME, engine=engine
) -> Set:
    # Get a list of URLs or all of them
    urls_query_txt = ", ".join([f"'{x}'" for x in urls])
    if urls:
        result = pd.read_sql_query(
            f"""SELECT DISTINCT "deck_url"
        FROM "{tname}" WHERE "deck_url" IN ({urls_query_txt})""",
            engine,
        )

    else:
        result = pd.read_sql_query(
            f"""SELECT DISTINCT "deck_url"
        FROM "{tname}""",
            engine,
        )

    return set(result["deck_url"])


@task
def check_if_all_cards_exist(
    deck_url: str, error_table: str = f"{MTGMETA_CARDS_TNAME}_errors", engine=engine
) -> str:
    """Returns the same input string or (find out what to map
    to failed state)

    Non mappable card slugs are saved to an error table
    """
    return deck_url


# https://docs.prefect.io/core/concepts/mapping.html#filter-map-output
filter_results = FilterTask(
    filter_func=lambda x: not isinstance(x, (BaseException, SKIP, type(None)))
)


@task
def transf_landing_to_decks(deck_url: str, engine=engine) -> pd.DataFrame:
    """Loads deck and cards_in_deck from mtgmeta.io tables and transform
    them into one dataframe suitable for decks table
    """
    return transformed_df


@task
def upsert_into_decks(
    df: pd.DataFrame, decks_table=DECKS_TNAME, engine=engine
) -> pd.DataFrame:
    """Upsert the transformed df to the decks table"""
    upsert_df(df, table_name=decks_table, engine=engine)

    return df


# Transform mtgmetaio tables data into canonical format
# Save them to decks table.
def get_flow_landing_to_decks(for_urls: List = []) -> Flow:
    """For each deck url, check if all cards exist in cards db
       if not, flag so we can correct it
       perform transformation
       and than upsert into decks table

    Args:
        for_urls (List, optional): Pass a list if you wish to execute the flow
            only for specif urls. Defaults to [] (process all urls in table).

    Returns:
        Flow: prefect flow, that can be run
    """
    with Flow("landing-to-decks") as flow_landing_to_decks:

        # Fetch specific list of urls or all
        deck_urls = get_list_or_all_deck_urls(urls=[])

        cards_checked = check_if_all_cards_exist.map(
            deck_urls
        )  # should output the same list of urls, if not failed
        unproblematic_decks = filter_results(cards_checked)
        transformed_df = transf_landing_to_decks.map(unproblematic_decks)
        upserted_into_decks = upsert_into_decks(df=transformed_df)

    return flow_landing_to_decks
