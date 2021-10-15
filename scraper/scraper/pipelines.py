# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import json
import logging

import jsonschema

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from jsonschema import validate
from scrapy.exceptions import DropItem
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .items import (
    CardsInDeck,
    Deck,
    VStats,
    RawItem,
    engine,
    mapper_registry,
    RAW_DECK_JSON_SCHEMA,
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
            raise DropItem(f"Raw Item did not validate: {item['deck_url']}")


class ParseToCorrectTypes:
    def open_spider(self, spider):
        self.file = open("items.jl", "w")

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict()) + "\n"
        self.file.write(line)
        return item


class ConvertToDataClassAndStore:
    def __init__(self):
        """
        Initializes database connection and sessionmaker
        Creates tables
        """
        mapper_registry.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)
