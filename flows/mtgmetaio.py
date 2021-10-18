from helpers.upsert_df import upsert_df
import logging
import os
from typing import List, Set

import pandas as pd
from sqlalchemy import create_engine, engine

import prefect
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
            f'scrapy crawl mtgmetaio -O "mtgmetaio-{datetime.utcnow():%Y-%m-%d-%H-%M}.json"'
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
    deck_url: str,
    error_table: str = f"{MTGMETA_CARDS_TNAME}_errors",
    engine=engine,
    mtgmeta_cards_tname: str = MTGMETA_CARDS_TNAME,
) -> str:
    """Returns the same input string or (find out what to map
    to failed state)

    Non mappable card slugs are saved to an error table
    """
    logger = prefect.context.get("mtgmetaio_check_if_all_cards_exist")

    query_mtgmeta_cards_ljoin_canonical_cards = f"""
    WITH filtered as (
        SELECT * FROM "{mtgmeta_cards_tname}"
        WHERE "deck_url" = '{deck_url}'
    )
    , result as (
        SELECT "deck_url", "card_slug", "name_slug" as "canonical_skryfall_name_slug"
        FROM filtered
        LEFT JOIN "cards"
        ON "card_slug"="name_slug"
    )
        SELECT *
        FROM result
        WHERE "canonical_skryfall_name_slug" is null
    """

    errors_df = pd.read_sql_query(query_mtgmeta_cards_ljoin_canonical_cards, engine)
    if not errors_df.empty:
        logger.info(
            f"Deck {deck_url} contains cards which are not in cards database. "
            f"They were saved in {error_table}. This deck will not be added to decks table."
        )
        upsert_df(errors_df, error_table, engine)
        return None

    return deck_url


# https://docs.prefect.io/core/concepts/mapping.html#filter-map-output
filter_results = FilterTask(
    filter_func=lambda x: not isinstance(x, (BaseException, SKIP, type(None)))
)


@task
def transf_landing_to_decks(
    deck_url: str,
    engine=engine,
    mtgmeta_cards_tname: str = MTGMETA_CARDS_TNAME,
    mtgmeta_decks_tname: str = MTGMETA_DECK_TNAME,
    decks_tname: str = DECKS_TNAME,
) -> pd.DataFrame:
    """Loads deck and cards_in_deck from mtgmeta.io tables and transform
    them into one dataframe suitable for decks table
    """
    logger = prefect.context.get("mtgmetaio_transf_landing_to_decks")

    query_mtgmeta_decks_join_canonical_cards = f"""
        WITH filtered_cards as (
            SELECT * FROM "{mtgmeta_cards_tname}"
            WHERE "deck_url" = '{deck_url}'
        )
        , filtered_decks as (
            SELECT "deck_url", "deckname" as "deck_name"
            FROM "{mtgmeta_decks_tname}"
            WHERE "deck_url" = '{deck_url}'
        )
        , joint_mtgmeta as (
            SELECT *
            FROM filtered_decks
            NATURAL JOIN filtered_cards
        )
        SELECT "deck_url" as "deck_id", "main", "quantity", "card_slug",
            "cards"."name" as "card_name", "deck_name"
        FROM joint_mtgmeta
        LEFT JOIN "cards"
        ON joint_mtgmeta."card_slug"="cards"."name_slug"
    """

    transformed_df = pd.read_sql_query(query_mtgmeta_decks_join_canonical_cards, engine)
    transformed_df["deck_name"] = transformed_df["deck_name"].fillna(
        transformed_df["deck_id"]
    )
    transformed_df["in"] = transformed_df["main"].apply(
        lambda x: "MAIN" if x else "SIDEBOARD"
    )
    transformed_df["for_explosion"] = transformed_df["quantity"].apply(
        lambda x: list(range(x))
    )
    transformed_df = transformed_df.explode("for_explosion", ignore_index=True)
    transformed_df = transformed_df.sort_values(by=["in", "card_name"]).reset_index()
    transformed_df["card_id_in_deck"] = transformed_df.index
    transformed_df = transformed_df[
        ["card_id_in_deck", "deck_id", "card_name", "in", "deck_name"]
    ]
    transformed_df = transformed_df.set_index(["card_id_in_deck", "deck_id"])

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
