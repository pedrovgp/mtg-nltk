# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import json
import logging
import re
from datetime import datetime

import jsonschema

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from jsonschema import validate
from scrapy.exceptions import DropItem
from slugify.slugify import slugify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dataclasses import fields

from .items import (
    RAW_DECK_JSON_SCHEMA,
    CardsInDeck,
    Deck,
    RawItem,
    VStats,
    engine,
    mapper_registry,
)

logger = logging.getLogger("mtgmetaio_pipelines")


def is_json_valid_against_schema(json_python_data, schema):
    try:
        validate(instance=json_python_data, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        logger.debug(err)
        return False
    return True


# Convert json to python object.
# jsonData = json.load(
#     open(
#         "/media/pv/DATA/pv/Documents/pv/projetos/mtg-nltk/data/mtgmetaio-2021-10-15.json",
#         "r",
#     )
# )


class StoreRaw:
    def __init__(self):
        """
        Initializes database connection and sessionmaker
        Creates tables
        """
        mapper_registry.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        """Save raw data in the database
        This method is called for every item pipeline component
        """
        session = self.Session()
        deck_url = item["deck_url"]
        raw_data = item
        item_db = RawItem(deck_url=deck_url, raw_data=raw_data)

        try:
            session.merge(item_db)
            session.commit()

        except Exception as e:
            logger.error("Commiting raw data failed")
            logger.error(e)
            session.rollback()
            raise e

        finally:
            session.close()

        return item


class ValidateRawSchema:
    def process_item(self, item, spider):
        if is_json_valid_against_schema(item, RAW_DECK_JSON_SCHEMA):
            return item
        else:
            logger.error(f"Raw Item did not validate: {item['deck_url']}")
            raise DropItem(f"Raw Item did not validate: {item['deck_url']}")


class ParseToCorrectTypes:
    int_pat = r"\d+"
    float_pat = r"\d+\.?\d+"
    short_date_pat = "\d+ \w{3} \d{4}"

    def process_item(self, item, spider):
        """Extracts and convret date to correspond to
        expected by items.py (Deck, VStats, CardsInDeck)"""

        deck_cleaned = {}

        deck_cleaned["deck_url"] = item["deck_url"]
        price = re.match(self.float_pat, item["price"])
        deck_cleaned["price"] = float(price.group()) if price else None
        metashare = re.match(self.float_pat, item["metashare"])
        deck_cleaned["metashare"] = (
            float(metashare.group()) / 100 if metashare else None
        )
        global_performance = re.match(self.float_pat, item["global_performance"])
        deck_cleaned["global_performance"] = (
            float(global_performance.group()) / 100 if global_performance else None
        )
        # extract date from era, break into two, and pop era
        era_begin, era_end = re.findall(self.short_date_pat, item["era"])
        deck_cleaned["era_begin"] = (
            datetime.strptime(era_begin, "%d %b %Y") if era_begin else None
        )
        deck_cleaned["era_end"] = (
            datetime.strptime(era_end, "%d %b %Y") if era_end else None
        )

        # Process each card in the decklist
        cleaned_cards = []
        for card in item["cards_in_deck"]:
            new_card = {}
            new_card["deck_url"] = item["deck_url"]
            new_card["main"] = bool(int(card.pop("data-main")))
            new_card["quantity"] = int(card.pop("data-qt"))
            new_card["card_name"] = card.pop("data-name")
            new_card["card_slug"] = slugify(new_card["card_name"], separator="_")
            cleaned_cards.append(new_card)
        deck_cleaned["cards_in_deck"] = cleaned_cards

        # Process each versus stat from the deck
        cleaned_vs = []
        for vs in item["vs_stats"]:
            new_vs = {}
            new_vs["deck_url"] = item["deck_url"]
            new_vs["vs_deck_url"] = vs.pop("vs_deck_url")
            matches = re.match(self.int_pat, vs.pop("matches"))
            new_vs["matches"] = int(matches.group()) if matches else 0
            new_vs["performance"] = float(
                re.match(self.float_pat, vs.pop("data-perf")).group()
            )
            cleaned_vs.append(new_vs)
        deck_cleaned["vs_stats"] = cleaned_vs

        logger.debug("DEBUGGING item cleaned:")
        logger.debug(deck_cleaned)
        return deck_cleaned


class ConvertToDataClassAndStore:
    def __init__(self):
        """
        Initializes database connection and sessionmaker
        Creates tables
        """
        mapper_registry.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        """Converts dict to dataclass and save to database
        This method is called for every item pipeline component
        """
        session = self.Session()

        def commit(session=session):
            try:
                session.commit()

            except Exception as e:
                logger.error("Commiting data failed")
                logger.error(e)
                session.rollback()
                raise e

        # Process each versus stat from the deck
        # for all deck_url and vs_deck_url in vs_stats
        # check if the corresponding deck exist
        # if not, create it
        # than add stats
        deck_urls_that_must_be_registered = set()
        vs_stats_to_register = []
        for vs in item["vs_stats"]:
            deck_urls_that_must_be_registered.add(vs["deck_url"])
            deck_urls_that_must_be_registered.add(vs["vs_deck_url"])
            vs_stats_to_register.append(VStats(**vs))

        decks_already_in_db = set(
            x[0]
            for x in session.query(Deck.deck_url)
            .filter(Deck.deck_url.in_(deck_urls_that_must_be_registered))
            .all()
        )
        decks_to_register = [
            Deck(deck_url)
            for deck_url in deck_urls_that_must_be_registered.difference(
                decks_already_in_db
            )
        ]

        session.add_all(decks_to_register)
        commit()
        for vs in vs_stats_to_register:
            # there is no merge_all (but we need merge here to update stats)
            session.merge(vs)
        commit()

        # Now process the cards in the deck
        for card in item["cards_in_deck"]:
            session.add(CardsInDeck(**card))
        commit()

        # Lastly, merge all info about the deck
        # If it was created only with pk, now other data will be added
        deck_only_dict = {
            k: v for k, v in item.items() if k not in ["cards_in_deck", "vs_stats"]
        }
        deck_to_merge = Deck(**deck_only_dict)
        logger.debug("logger.debug(deck_to_merge)")
        logger.debug(deck_to_merge)
        session.merge(deck_to_merge)
        commit()

        session.close()

        return item
