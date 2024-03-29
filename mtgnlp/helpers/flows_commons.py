from mtgnlp import config
import logging
from typing import List

import pandas as pd
from sqlalchemy import create_engine, engine

logger = logging.getLogger()

engine = create_engine(config.DB_STR)


def get_list_or_all_deck_ids(
    deck_ids: List = [], tname: str = config.CARDS_TNAME, engine=engine
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
