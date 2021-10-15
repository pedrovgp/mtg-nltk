# Scrapes https://mtgmeta.io for deck data and stats

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import (
    LinkExtractor,
)  # https://docs.scrapy.org/en/latest/topics/link-extractors.html#topics-link-extractors
from scrapy import Selector
from itemadapter import ItemAdapter

# TODO add items and
# Then, add middleware to split item in multiple items (https://docs.scrapy.org/en/latest/faq.html#how-to-split-an-item-into-multiple-items-in-an-item-pipeline)
class SplitDeckItemsMiddleware:
    def process_spider_output(self, response, result, spider):
        for deck_item in result:
            yield deck_item
            # deck_adapter = ItemAdapter(deck_item)
            # yield deck_adapter
            #         adapter = ItemAdapter(item)
            #         for _ in range(adapter["multiply_by"]):


class MtgMetaIoSpider(CrawlSpider):
    name = "mtgmetaio"
    pat_of_deck_urls = r"https://mtgmeta.io/decks/\d+"
    custom_settings = {
        "SPIDER_MIDDLEWARES": {
            "scraper.spiders.mtgmetaio.SplitDeckItemsMiddleware": 543,
        },
        "DEPTH_LIMIT": 1,  # 0 means no limit
        "ITEM_PIPELINES": {
            # 'mybot.pipelines.validate.ValidateMyItem': 300,
            # 'mybot.pipelines.validate.StoreMyItem': 800,
        },
    }
    start_urls = [
        "https://mtgmeta.io",
        "https://mtgmeta.io/decks",
        "https://mtgmeta.io/decks?f=standard",
        "https://mtgmeta.io/decks?f=pioneer",
        "https://mtgmeta.io/decks?f=modern",
        "https://mtgmeta.io/decks?f=legacy",
        "https://mtgmeta.io/decks?f=pauper",
        "https://mtgmeta.io/decks?f=historic",
        "https://mtgmeta.io/decks/23910",
    ]
    rules = [
        # Rule to extract deck info AND follow deck urls
        Rule(
            link_extractor=LinkExtractor(allow=r"https://mtgmeta.io/decks/\d+"),
            callback="parse_deck_url",
            follow=True,
        ),
        Rule(
            link_extractor=LinkExtractor(allow=r"https://mtgmeta.io/.*"),
            follow=True,
        ),
    ]

    def parse_deck_url(self, response):

        self.logger.info(f"Deck parsing: {response.url}")

        result_to_export = {}

        # deck identifiers
        result_to_export["deckname"] = response.selector.xpath(
            '//h1[contains(@class, "deckname")]/text()'
        ).get()
        result_to_export["deck_url"] = response.url

        # Global stats showing below the deck name, right on top of the page
        deck_global_stats = response.selector.xpath('//ul[@id="deckstats"]/li/text()')
        # Sample deck_global_stats
        #  ['standard',
        #  '149.68$\n\n',
        #  '5.16% Metashare',
        #  '58.5% [0% - 0%] Global Performance',
        #  '16 Sep 2021 - 14 Oct 2021']
        result_to_export.update(
            {
                "format": deck_global_stats[0].get(),
                "price": deck_global_stats[1].get(),
                "metashare": deck_global_stats[2].get(),
                "global_performance": deck_global_stats[3].get(),
                "era": deck_global_stats[4].get(),
            }
        )

        # Cards in deck
        decklist = response.selector.xpath(
            '//ul[contains(@class, "sampledecklist")]/li'
        )
        # cards_in_deck is a list of dicts
        result_to_export["cards_in_deck"] = [
            b.attrib for b in decklist if b.attrib.get("data-name", False)
        ]
        # sample cards_in_deck
        # [{'data-qt': '2',  # amount of this card in deck
        # 'data-main': '0',  # 0: sideboard, 1: main deck
        # 'data-name': 'portable hole', # card name
        # 'data-edition': 'afr',
        # 'data-number': '33',
        # 'data-cmc': '1'}]

        # deck identifiers
        deckvs = response.selector.xpath(
            '//ul[contains(@class, "deckvs")]/li[@data-perf]'
        )
        vs_results = []
        for d in deckvs:
            d_selector = Selector(text=d.get())
            attrib_dict = d.attrib
            attrib_dict["matches"] = d_selector.xpath(
                '//span[contains(@class, "matches")]/text()'
            ).get()
            attrib_dict["vs_deck_url"] = d_selector.xpath(
                '//a[has-class("btn")]/@href'
            ).get()
            vs_results.append(attrib_dict)
        # sample vs_results
        # [{'data-pos': '70',  # no idea o meaning
        # 'data-perf': '33.3',  # percentage of times this deck won against vs_deck_url
        # 'data-name': 'temur epiphany',  # vs_deck_url name
        # 'class': 'hidden',  # useless
        # 'matches': '(3)',  # number of maches they played against each other
        # 'vs_deck_url': 'https://mtgmeta.io/decks/23897',
        # }]
        result_to_export["vs_results"] = vs_results

        # self.log(f'Result: {result_to_export}')

        yield result_to_export
