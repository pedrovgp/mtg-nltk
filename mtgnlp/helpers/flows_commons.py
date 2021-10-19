from helpers.upsert_df import upsert_df
import logging
import os
from typing import List

import pandas as pd
from sqlalchemy import create_engine, engine

import prefect
from prefect import Flow, Parameter, Task, task

from prefect.tasks.control_flow.filter import FilterTask
from prefect.engine.signals import SKIP


logger = logging.getLogger()

engine = create_engine(os.getenv("DB_STR"))
MTGMETA_DECK_TNAME = os.getenv("MTGMETA_DECK_TNAME")
MTGMETA_CARDS_TNAME = os.getenv("MTGMETA_CARDS_TNAME")
CARDS_TNAME = os.getenv("CARDS_TNAME")
DECKS_TNAME = os.getenv("DECKS_TNAME")


def get_list_or_all_deck_ids(
    deck_ids: List = [], tname: str = CARDS_TNAME, engine=engine
) -> List:
    # Get a list of URLs or all of them
    urls_query_txt = ", ".join([f"'{x}'" for x in deck_ids])
    if deck_ids:
        result = pd.read_sql_query(
            f"""SELECT DISTINCT "deck_id"
        FROM "{tname}" WHERE "deck_id" IN ({urls_query_txt})""",
            engine,
        )

    else:
        result = pd.read_sql_query(
            f"""SELECT DISTINCT "deck_id"
                FROM "{tname}"
            """,
            engine,
        )

    return list(result["deck_id"])
