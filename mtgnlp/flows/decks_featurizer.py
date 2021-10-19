from mtgnlp import config
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

from mtgnlp.helpers.flows_commons import get_list_or_all_deck_ids

logger = logging.getLogger()

engine = create_engine(config.DB_STR)
MTGMETA_DECK_TNAME = config.MTGMETA_DECK_TNAME
MTGMETA_CARDS_TNAME = config.MTGMETA_CARDS_TNAME
DECKS_TNAME = config.DECKS_TNAME
DECKS_FEATURES_TNAME = config.DECKS_FEATURES_TNAME

# Creates a deck_features table and adds all calculated features to it
def get_flow_featurize_decks(for_urls: List = []) -> Flow:
    """For each deck url,
    create individual cards features
    sinergy features
    antergy features

    Args:
        for_urls (List, optional): Pass a list if you wish to execute the flow
            only for specif urls. Defaults to [] (process all urls in table).

    Returns:
        Flow: prefect flow, that can be run
    """
    with Flow("featurize_decks") as featurize_decks:

        test_ids = [
            "https://mtgmeta.io/decks/20311?rid=275510",  # all cards in cards
        ]
        # Fetch specific list of urls or all
        deck_urls = get_list_or_all_deck_ids(deck_ids=[])
        # deck_urls = get_list_or_all_deck_urls(urls=test_urls)

        cards_features = create_cards_features.map(deck_urls)
        sinergy_features = create_sinergy_features.map(deck_urls)
        antergy_features = create_antergy_features.map(deck_urls)

    return featurize_decks


if __name__ == "__main__":
    flow = get_flow_featurize_decks()
    flow.visualize()
    flow_state = flow.run()
    flow.visualize(flow_state=flow_state)
