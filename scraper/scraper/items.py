# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from dataclasses import dataclass, field
from datetime import datetime
from os import getenv
from typing import Dict, List

from scrapy import Item
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    JSON,
)
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import registry, relationship

mapper_registry = registry()
engine = create_engine(getenv("DB_STR"))

# define all the items I need as DataClass items
# decks table (deck_url/url, name, format, metashare, global_performance, era, price)
# decks cards table (deck_url/url, card_name, card_slug, quantity, main)
# vs_stats table (deck_url/url, vs_deck_url/url, performance, matches)
CARDS_IN_DECK_TNAME = "mtgmetaio_cards_in_deck"
DECK_TNAME = "mtgmetaio_decks"
VSTATS_TNAME = "mtgmetaio_vstats"
RAW_ITEM_TNAME = "mtgmetaio_rawitems"

# Validate raw data against expected schema
RAW_DECK_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "deckname": {"type": "string"},
            "deck_url": {"type": "string"},
            "format": {"type": "string"},
            "price": {"type": "string"},
            "metashare": {"type": "string"},
            "global_performance": {"type": "string"},
            "era": {"type": "string"},
            "cards_in_deck": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "data-qt": {"type": "string"},
                        "data-main": {"type": "string"},
                        "data-name": {"type": "string"},
                        "data-edition": {"type": "string"},
                        "data-number": {"type": "string"},
                        "data-cmc": {"type": "string"},
                    },
                    "required": [
                        "data-main",
                        "data-name",
                        "data-qt",
                    ],
                },
            },
            "vs_stats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "data-pos": {"type": "string"},
                        "data-perf": {"type": "string"},
                        "data-name": {"type": "string"},
                        "class": {"type": "string"},
                        "matches": {"type": "string"},
                        "vs_deck_url": {"type": "string"},
                    },
                    "required": [
                        "data-perf",
                        "matches",
                        "vs_deck_url",
                    ],
                },
            },
        },
        "required": [
            "cards_in_deck",
            "deck_url",
            "vs_stats",
        ],
    },
}


@mapper_registry.mapped
@dataclass
class CardsInDeck:
    __tablename__ = CARDS_IN_DECK_TNAME
    __sa_dataclass_metadata_key__ = "sa"
    card_name: str = field(metadata={"sa": Column(String(100), primary_key=True)})
    deck_url: str = field(
        default=None,
        metadata={"sa": Column(ForeignKey(f"{DECK_TNAME}.deck_url"), primary_key=True)},
    )
    main: bool = field(
        default=True, metadata={"sa": Column(Boolean(), primary_key=True)}
    )
    quantity: int = field(
        default=None, metadata={"sa": Column(Integer, nullable=False)}
    )
    card_slug: str = field(
        default=None, metadata={"sa": Column(String(100), nullable=False)}
    )


@mapper_registry.mapped
@dataclass
class VStats:
    __tablename__ = VSTATS_TNAME
    __sa_dataclass_metadata_key__ = "sa"
    deck_url: str = field(
        default=None,
        metadata={"sa": Column(ForeignKey(f"{DECK_TNAME}.deck_url"), primary_key=True)},
    )
    vs_deck_url: str = field(
        default=None,
        metadata={"sa": Column(ForeignKey(f"{DECK_TNAME}.deck_url"), primary_key=True)},
    )
    matches: int = field(default=None, metadata={"sa": Column(Integer, nullable=False)})
    performance: float = field(
        default=None, metadata={"sa": Column(Float(5), nullable=False)}
    )


@mapper_registry.mapped
@dataclass
class Deck:
    __tablename__ = DECK_TNAME
    __sa_dataclass_metadata_key__ = "sa"

    deck_url: str = field(metadata={"sa": Column(String, primary_key=True)})
    deckname: str = field(default=None, metadata={"sa": Column(String(200))})
    format: str = field(default=None, metadata={"sa": Column(String(50))})
    metashare: float = field(default=None, metadata={"sa": Column(Float(5))})
    global_performance: float = field(default=None, metadata={"sa": Column(Float(5))})
    era_begin: datetime = field(default=None, metadata={"sa": Column(DateTime())})
    era_end: datetime = field(default=None, metadata={"sa": Column(DateTime())})
    price: float = field(default=None, metadata={"sa": Column(Float(10))})
    # cards_in_deck: List[CardsInDeck] = field(
    #     default_factory=list, metadata={"sa": relationship("CardsInDeck")}
    # )
    # vs_stats: List[VStats] = field(
    #     default_factory=list,
    #     metadata={"sa": relationship("VStats", foreign_keys="VStats.deck_url")},
    # )


@mapper_registry.mapped
@dataclass
class RawItem:
    __tablename__ = RAW_ITEM_TNAME
    __sa_dataclass_metadata_key__ = "sa"

    deck_url: str = field(metadata={"sa": Column(String, primary_key=True)})
    raw_data: Dict = field(metadata={"sa": Column(JSON, nullable=False)})


if __name__ == "__main__":
    # mapper_registry.metadata.create_all(engine)
    mapper_registry.metadata.drop_all(engine)
    pass
