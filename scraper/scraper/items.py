# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy import Item

from dataclasses import dataclass
from dataclasses import field
from typing import List
from datetime import datetime

from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import String, Integer, Float, DateTime, Boolean
from sqlalchemy.orm import registry
from sqlalchemy.orm import relationship
from sqlalchemy import MetaData

# from .settings import ENGINE

mapper_registry = registry()

# TODO define all the items I need as DataClass items
# decks table (deck_id/url, name, format, metashare, global_performance, era, price)
# decks cards table (deck_id/url, card_name, card_slug, quantity, main)
# vs_stats table (deck_id/url, vs_deck_id/url, performance, matches)
CARDS_IN_DECK_TNAME = "mtgmetaio_cards_in_deck"
DECK_TNAME = "mtgmetaio_decks"
VSTATS_TNAME = "mtgmetaio_vstats"


@mapper_registry.mapped
@dataclass
class CardsInDeck:
    __tablename__ = CARDS_IN_DECK_TNAME
    __sa_dataclass_metadata_key__ = "sa"
    id: int = field(init=False, metadata={"sa": Column(Integer, primary_key=True)})
    deck_url: str = field(
        default=None,
        metadata={"sa": Column(ForeignKey(f"{DECK_TNAME}.deck_url"), nullable=False)},
    )
    main: bool = field(default=True, metadata={"sa": Column(Boolean(), nullable=False)})
    quantity: int = field(
        default=None, metadata={"sa": Column(Integer, nullable=False)}
    )
    card_name: str = field(
        default=None, metadata={"sa": Column(String(100), nullable=False)}
    )
    card_slug: str = field(
        default=None, metadata={"sa": Column(String(100), nullable=False)}
    )


@mapper_registry.mapped
@dataclass
class VStats:
    __tablename__ = VSTATS_TNAME
    __sa_dataclass_metadata_key__ = "sa"
    id: int = field(init=False, metadata={"sa": Column(Integer, primary_key=True)})
    deck_url: str = field(
        default=None,
        metadata={"sa": Column(ForeignKey(f"{DECK_TNAME}.deck_url"), nullable=False)},
    )
    vs_deck_url: str = field(
        default=None,
        metadata={"sa": Column(ForeignKey(f"{DECK_TNAME}.deck_url"), nullable=False)},
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

    deck_url: str = field(
        default=None, metadata={"sa": Column(String, primary_key=True)}
    )
    name: str = field(default=None, metadata={"sa": Column(String(200))})
    format: str = field(default=None, metadata={"sa": Column(String(50))})
    metashare: float = field(default=None, metadata={"sa": Column(Float(5))})
    global_performance: float = field(
        default=None, metadata={"sa": Column(Float(5), nullable=False)}
    )
    era_begin: datetime = field(default=None, metadata={"sa": Column(DateTime())})
    era_end: datetime = field(default=None, metadata={"sa": Column(DateTime())})
    price: float = field(default=None, metadata={"sa": Column(Float(10))})
    cards_in_deck: List[CardsInDeck] = field(
        default_factory=list, metadata={"sa": relationship("CardsInDeck")}
    )
    vs_stats: List[VStats] = field(
        default_factory=list, metadata={"sa": relationship("VsStats")}
    )


if __name__ == "__main__":
    # mapper_registry.metadata.create_all(ENGINE)
    # mapper_registry.metadata.drop_all(ENGINE)
    pass
