# # Generate a deck from a given set of cards

# Partly following the idea outlined here:
# https://hackernoon.com/machine-learning-magic-64a6a7f864d4
#
# 1. Determine a system of points for cards and interactions given by deck graph
# 1.1 Suggestions: P/T by CMC, P/T by coloured CMC, number of parts, entities affected,
#     verbs by cost, point for different verbs, etc...
# 1.2 Also: it could be made relative to other cards in the set, for example,
#     How are the cards in the deck affected by text in cards of the whole set?
#     How does these points compare to average points of the rest of the set?
# 2. Define a fitness function as a function of these points
# 2.1 There is room here for human intervention assigning higher weights to
#     whatever she/he wants to favor (speed: lower CMCs, surprises: instants, etc.)
# 3. Use GA algorithms to generate a deck with the given fitness function
#
# NEXT
# 4. Don't know yet

# **DESIRED RESULT**:
# Given a set of cards, a deck list with cards within that set.

import pandas as pd
import numpy as np
import re

import functools

import streamlit as st

import logging
import inspect
import linecache
import os

st.header('MTG: deck geration')
st.write('''Welcome! This app uses genetic algorithms to generate MTG decks based on your given
optimization criteria. It generates X decks, evaluates and uses the best from X decks to 
generate a next generation of X decks. You can choose how many generations you want to evolve your deck.
''')

try:
    if __file__: pass
except NameError:
    # for running in ipython
    fname = '04c_ga_for_deck_building.py'
    __file__ = os.path.abspath(os.path.realpath(fname))

logPathFileName = './logs/' + '04c.log'

# create logger'
logger = logging.getLogger('04c')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{logPathFileName}", mode='w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# This is for jupyter notebook
# from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()
# This is for terminal
from tqdm import tqdm
tqdm.pandas(desc="Progress")

# # Params

from sqlalchemy import create_engine
import sqlalchemy

engine = create_engine('postgresql+psycopg2://mtg:mtg@localhost:5432/mtg')
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
engine.connect()

INITIAL_SET = 'TMP'

st.write(f'At first, we are considering only {INITIAL_SET} as a collection to build the deck from.')

df_tempest = pd.read_sql_query(f'''
   SELECT * 
   FROM cards
   WHERE cards.set IN ('{INITIAL_SET}')''',
                               engine,
                               # index_col='id'
                               )
df_tempest = pd.concat([df_tempest for x in range(4)]).reset_index(drop=True)
logger.info(f'df_tempest shape: {df_tempest.shape}')

# initial number of cards in the decks
IND_INIT_SIZE = 36
# min and max items should be the max deck size (in cards)
MIN_ITEM, MAX_ITEM = st.sidebar.slider(
    'Minimum and maximum cards for deck',
    min_value=30, max_value=40, value=(34, 36)
) or (34, 36)
# NBR_ITEMS would be the whole pool of cards we can build our deck from
NBR_ITEMS = df_tempest.shape[0]

# lets start only by maximizing P/T, both weighting 1
POWER_WEIGHT = st.sidebar.number_input(
    'How much would you like Power to weight in optimization?',
    min_value=1.0, max_value=10.0, value=1.0
)
TOUGH_WEIGHT = st.sidebar.number_input(
    'How much would you like Thoughness to weight in optimization?',
    min_value=1.0, max_value=10.0, value=1.0
)

logger.info(f'Class DeckDf creation')


class DeckVal:
    """This is a class for holding a deck as a dataframe
    but also implementing methods for metrics calculations
    and plottings"""

    def __init__(self, df,
                 max_cards=MAX_ITEM,
                 min_cards=MIN_ITEM,
                 *args, **kwargs
                 ):
        self.df = df
        self.cmc = self.df['cmc']
        self.power_sum = self.df['power_num'].sum()
        self.toughness_sum = self.df['toughness_num'].sum()
        self.total_cards = self.df.shape[0]
        self.max_cards = max_cards
        self.min_cards = min_cards
        self.colors_set = self.get_colors_set()
        self.colors_count = len(self.colors_set)

    # from: https://mtg.gamepedia.com/Mana_curve
    ideal_mana_curve = pd.DataFrame([9, 8, 8, 11])
    ideal_mana_curve.index = ideal_mana_curve.index + 1

    WEIGHTS = (POWER_WEIGHT, TOUGH_WEIGHT, -22500.0)

    def valuation(self):
        """Returns a tuple of metrics for deck valuation"""

        # Ensure deck card limits problems are dominated
        if self.total_cards > MAX_ITEM or self.total_cards < MIN_ITEM:
            return tuple(-10000 for x in range(len(self.WEIGHTS)))

        result = (
            self.power_sum,
            self.toughness_sum,
            self.colors_count
        )
        if not len(result) == len(self.WEIGHTS):
            raise Exception(f'Results and WEIGHTS must have the same length.')

        return result

    def get_colors_set(self):

        self.df['colors_set'] = self.df['colors'].apply(lambda x: set(re.findall('([A-Z]){1}', x)))
        self.colors_set = functools.reduce(lambda a, b: a.union(b), self.df['colors_set'].values, set())
        return self.colors_set

    def compute_deck_stats(self):
        deck_df = self.df

        st.write(f'Final deck has {deck_df.shape[0]} cards')

        st.write('Mana curve')
        cmc = deck_df.pivot_table(
            index=['cmc'], values=['id'], aggfunc=lambda x: x.count()
        ).transpose()
        logger.info(cmc)
        st.write(cmc)

        st.write('Ideal mana curve')
        st.write(self.ideal_mana_curve.transpose())

        st.write('Cards by color')
        colors = deck_df.pivot_table(
            index=['colors'], values=['id'], aggfunc=lambda x: x.count()
        ).transpose()
        logger.info(colors)
        st.write(colors)

        st.write('Cards that add mana')
        has_add = deck_df.pivot_table(
            index=['has_add'], values=['id'], aggfunc=lambda x: x.count()
        ).transpose()
        logger.info(has_add)
        st.write(has_add)

        st.write('Cards by type')
        types = deck_df.pivot_table(
            index=['types'], values=['id'], aggfunc=lambda x: x.count()
        ).transpose()
        logger.info(types)
        st.write(types)

        st.write('Power and toughness by type')
        pt_by_type = deck_df.pivot_table(
            index=['types'], values=['power_num', 'toughness_num'], aggfunc=np.sum
        )
        logger.info(pt_by_type)
        st.write(pt_by_type)

        st.write('Power and toughness by color')
        pt_by_color = deck_df.pivot_table(
            index=['colors'], values=['power_num', 'toughness_num'], aggfunc=np.sum
        )
        logger.info(pt_by_color)
        st.write(pt_by_color)

        st.write('Full deck list')
        deck = (
            deck_df.pivot_table(
                index=['name'], values=['id'], aggfunc=lambda x: x.count()
            )
                .merge(
                deck_df.pivot_table(
                    index=['name'], values=['power_num', 'toughness_num'], aggfunc=np.sum
                ), left_index=True, right_index=True

            )
        )

        logger.info(deck)
        st.write(deck)

        logger.info(f'{self.colors_count} color: {self.colors_set}')

def get_deck_val(set_of_ids, df = df_tempest):
    return DeckVal(df[df.index.isin(set_of_ids)])

#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

# References:
# https://deap.readthedocs.io/en/master/examples/ga_knapsack.html
# https://en.wikipedia.org/wiki/Knapsack_problem

import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# To assure reproductibility, the RNG seed is set prior to the items
# dict initialization. It is also seeded in main().
random.seed(64)

# Create the item dictionary: item name is an integer, and value is
# a (weight, value) 2-uple.
items = {}
# Create random items and store them in the items' dictionary.
# An item has a (weight, value) attribute
# NBR_ITEMS should be all cards of initial set/collection
for i in range(NBR_ITEMS):
    # 1 item would be one card, with all its attributes for later weighting
    items[i] = df_tempest.iloc[i]
    # items[i] = (random.randint(1, 10), random.uniform(0, 100))  # old

creator.create("Fitness", base.Fitness, weights=DeckVal.WEIGHTS)
# Individual is a backpack/deck (a set of items/cards)
# this stays the same (set is ok if copies of cards get different ids.
# If not, we need list so that a card can appear more than once in a deck)
creator.create("Deck", set, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Attribute generator
# this randomly generates items/cards
toolbox.register("attr_item", random.randrange, NBR_ITEMS)

# Structure initializers
# this repeats card pulling until we get a deck
toolbox.register("deck", tools.initRepeat, creator.Deck,
                 toolbox.attr_item, IND_INIT_SIZE)
# this creates a population of decks to be selected on best fitness
toolbox.register("population", tools.initRepeat, list, toolbox.deck)


# this evaluation function should carry our criteria for a good deck (start simple with P/T)
# it should return something of the same length as weights in Fitness above
def evalDeck(deck):
    d = get_deck_val(deck)
    return d.valuation()  # weight, value


# TODO crossover strategy requires deep thought
def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)  # Used in order to keep type
    ind1 &= ind2  # Intersection (inplace)
    ind2 ^= temp  # Symmetric Difference (inplace)
    return ind1, ind2

# Testing if adding black ony ids works
# black_ids = list(set(df_tempest[df_tempest['colors'] == '{B}'].index))
# TODO mutation strategy requires deep thought
def mutSet(deck):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(deck) > 0:  # We cannot pop from an empty set
            deck.remove(random.choice(sorted(tuple(deck))))
    else:
        deck.add(random.randrange(NBR_ITEMS))
        # deck.add(random.choice(black_ids))
    while len(deck) < MIN_ITEM:
        deck.add(random.randrange(NBR_ITEMS))
        # deck.add(random.choice(black_ids))
    while len(deck) > MAX_ITEM:
        deck.remove(random.choice(sorted(tuple(deck))))
    return deck,


toolbox.register("evaluate", evalDeck)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)


def main():
    random.seed(64)
    NGEN = st.sidebar.slider('How many deck generations to evolve?',
                             min_value=3, max_value=100, value=30) or 30
    MU = st.sidebar.slider('How many decks should a generation have?',
                           min_value=2, max_value=50, value=15) or 15
    LAMBDA = 100
    CXPB = 0.8
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    with st.spinner('Simulating, evolving, mutating, changing, procreating...'):
        algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                                  halloffame=hof)
        st.success('Simulation finished!!')
        # st.balloons()

    best_deck_list = list(hof[0])  # list of cards in best deck

    df_best_deck = get_deck_val(best_deck_list)
    df_best_deck.compute_deck_stats()

    return pop, stats, hof


if __name__ == "__main__":
    main()

    logger.info('Results available')

    logger.info(f'FINISHED: {__file__}')
