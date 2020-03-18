# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python [conda env:mtg]
#     language: python
#     name: conda-env-mtg-py
# ---

# # ETL of all cards in database

# The idea here is to
#
# 1. Get the cards json, build a dataframe of it
# 2. Prework a few things and split paragraphs to detect abilities, activated abilites/costs and effects.
# 3. Additionaly, detect condition and intensifier parts of a paragraph.
# 4. Use spacy to parse the cards text and identify entities.
# 5. Build the outgoing nodes and edges for each card (card -> text -> entities)
# 6. Build the incoming nodes and edges for a card (entities (cards attributes) -> card)
#
# Next:
# 7. Build the graphs for each card
#
# At the end, store it in a pickle to avoid parsing everything again next time, which takes a long time.
#
# **DESIRED RESULT**:
# outgoing_nodes_df = nodes for cards, text, tokens, entities
# outgoing_edges_df = edges from the nodes above
# incoming_nodes_df = nodes for cards and attribute entities
# incoming_edges_df = edges from the nodes above

import json

import networkx as nx
import pandas as pd
import re
from collections import defaultdict
from IPython.display import clear_output

import logging
import inspect
import linecache
import os

try:
    __file__
except NameError:
    # for running in ipython
    fname = '01.02-cards-ETL-token-nodes-edges-tables.py'
    __file__ = os.path.abspath(os.path.realpath(fname))

logPathFileName = './logs/' + '01.02.log'

# create logger'
logger = logging.getLogger('01.01')
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

def log_next_line(lines=1, level=logger.info):
    '''
    TODO: this will always log again this same function when called

    :param lines: how many relative lines ahead to log (negative for previous)
    :param level: logger function with log level to log
    :return: None, but logs stuff
    '''
    for i in range(1, lines+1):
        level(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + i))

# This is for jupyter notebook
# from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()
# This is for terminal
from tqdm import tqdm
tqdm.pandas(desc="Progress")

# + hideCode=false
sets = json.load(open('./AllSets.json', 'rb'))
# -

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
cards_all_sets=[]
for k, sett in sets.items():
    if (k in ['UGL', 'UST', 'UNH']) or (len(k)>3): # Ignore Unglued, Unstable and promotional things
        continue
    for card in sett['cards']:
        card['set'] = k
    cards_all_sets.extend(sett['cards'])    

# + hideCode=false
printings = json.load(open('./AllPrintings.json', 'rb'))
# -

cards_all=[]
for k, sett in printings.items():
    if (k in ['UGL', 'UST', 'UNH']) or (len(k)>3): # Ignore Unglued, Unstable and promotional things
        continue
    for card in sett['cards']:
        card['set'] = k
    cards_all.extend(sett['cards'])    

pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 101)

# # Params

ASPAS_TEXT = "ASPAS_TEXT"

mains_col_names = ['name', 'manaCost', 'text_preworked', 'type', 'power', 'toughness',
                   'types', 'supertypes', 'subtypes']


logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://mtg:mtg@localhost:5432/mtg')
engine.connect()

cards_text_parts_table_name = 'cards_text_parts'

# +
out_nodes_table_name = 'outnodes'
out_edges_table_name = 'outedges'

in_nodes_table_name = 'innodes'
in_edges_table_name = 'inedges'

# + [markdown] hideCode=true
# # Helping functions

# + code_folding=[1]
import collections
def splitDataFrameList(df,target_column,separator=None):
    '''
    https://gist.github.com/jlln/338b4b0b55bd6984f883
    df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column]#.split(separator)
        if isinstance(split_row, collections.Iterable):
            for s in split_row:
                new_row = row.to_dict()
                new_row[target_column] = s
                row_accumulator.append(new_row)
        else:
            new_row = row.to_dict()
            new_row[target_column] = pd.np.nan
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows, axis=1, args=(new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


# -

# # Create dataframe of cards
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
cards = cards_all
cards_df = pd.DataFrame.from_dict(cards)
cards_df = cards_df.drop_duplicates(subset=['name'])
cards_df = cards_df.drop(columns=['foreignData', 'legalities', 'prices', 'purchaseUrls', 
                                  'rulings', 'leadershipSkills'], errors='ignore')
# cards_df = cards_df.sample(200)

cards_df['id'] = cards_df['uuid']
cards_df = cards_df.set_index('id')

# ## Remove anything between parenthesis and replace name by SELF

# +
# Replace name by SELF and remove anything between parethesis
pattern_parenthesis = r' ?\(.*?\)'
def prework_text(card):
    t = str(card['text']).replace(card['name'], 'SELF')
    t = re.sub(pattern_parenthesis, '', t)
    return t

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe())+1))
cards_df['text_preworked'] = cards_df.apply(prework_text, axis=1)
#cards_df['text_preworked']

# +
# Set land text, which may be empty
def add_mana_text(text, sym):
    if not text:
        return '{T}: Add ' + sym +'.'
    elif '{T}: Add ' + sym not in text:
        return text + '\n' + '{T}: Add ' + sym +'.'
    return text

lands = [('Plains', '{W}'), ('Swamp', '{B}'), ('Island', '{U}'), ('Mountain', '{R}'), ('Forest', '{G}')]
for land_name, sym in lands:
    cards_df['text_preworked'] = cards_df.progress_apply(lambda x:
                                      add_mana_text(x['text_preworked'], sym)
                                      if isinstance(x['subtypes'], list) and land_name in x['subtypes']
                                      else x['text_preworked'],
                              axis=1
                             )
# -

# Check whether card can add mana
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
cards_df['has_add'] = cards_df['text_preworked'].apply(
    lambda x: True
              if re.findall(r'add ', str(x), flags=re.IGNORECASE)
              else False
)

sep = "ª"
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
if cards_df['text_preworked'].str.contains(sep).any():
    raise Exception("Bad separator symbol. It is contained in some text.")

assert cards_df[cards_df['text_preworked'].str.contains('\(').fillna(False)]['text_preworked'].empty

# # Domain specific vocabulary

# Let's build some domain specific vocabulary for MTG. For example, let's list supertypes, types, subtypes, know all card names, this kind f thing.

# Create set of cards names
cards_names = set(cards_df.name.unique())

# +
# Create set of supertypes
array_of_supertypes_tuples = cards_df['supertypes'].dropna().apply(tuple).unique()
cards_supertypes = tuple()
for tup in array_of_supertypes_tuples:
    cards_supertypes += tup
    
cards_supertypes = set(cards_supertypes)
cards_supertypes

# +
# Create set of types
array_of_types_tuples = cards_df['types'].dropna().apply(tuple).unique()
cards_types = tuple()
for tup in array_of_types_tuples:
    cards_types += tup
    
cards_types = set(cards_types)
#cards_types

# +
# Create set of types
array_of_subtypes_tuples = cards_df['subtypes'].dropna().apply(tuple).unique()
cards_subtypes = tuple()
for tup in array_of_subtypes_tuples:
    cards_subtypes += tup
    
cards_subtypes = set(cards_subtypes)
#cards_subtypes

# +
#cards_df.head(10).transpose()

# + deletable=false editable=false run_control={"frozen": true}
# import requests
# import pickle
# r = requests.get('http://media.wizards.com/2018/downloads/MagicCompRules%2020180713.txt')
# if not r.status_code == 200:
#     r.raise_for_status()
# comprules = r.text
# -
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
with open('rules.txt', 'r', encoding='latin-1') as f:
    comprules = '\n'.join(f.readlines())

kw_abilities_pat = r'702\.\d+\. ([A-Za-z ]+)'
abilities = re.findall(kw_abilities_pat, comprules, re.IGNORECASE)
abilities.pop(0) # Its just the rulings 
abilities.sort()
#abilities

# ## Load cards_df_pop_parts

# + code_folding=[]
# Lets avoid creating a pop node
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
cards_df_pop_parts = pd.read_sql_table(cards_text_parts_table_name, engine)
# -

# ### Checkings

# + deletable=false editable=false run_control={"frozen": true}
# (cards_df_pop_parts==cards_df_pop_parts2).all().all()

# + deletable=false editable=false run_control={"frozen": true}
# cards_df_pop_parts[cards_df_pop_parts['part_type']=='step_condition']['part'].unique()
# -

# ## Detecting special symbols

import itertools

patt = r'\{.*?\}'
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
t = cards_df_pop_parts['pop'].apply(lambda x: re.findall(patt, str(x))
                             if re.findall(patt, str(x)) else pd.np.nan)
symbols_set=set(itertools.chain.from_iterable(t.dropna()))
#symbols_set

weird_symbols = []
worth_ignoring = []#'{hr}','{½}','{∞}'] # Unglued or similar
# worth_ignoring.append('{CHAOS}')
symbols_explanation = {
    '{S}': {'explanation': 'Snow mana', 'example_card': 'Glacial Plating'},
    '{R/P}': {'explanation': 'can be paid with either {R} or 2 life', 'example_card': 'Rage Extractor'},
    '{Q}': {'explanation': '{Q} is the untap symbol', 'example_card': 'Order of Whiteclay'},
    '{E}': {'explanation': 'Energy counter', 'example_card': 'Consulate Surveillance'},
    '{C}': {'explanation': 'Colorless mana', 'example_card': 'Skarrg, the Rage Pits'},
    '{CHAOS}': {'explanation': 'It is only in Plane cards and for a specific kind of game',
                'example_card': 'Glimmervoid Basin'},
}
weird_cards = []

for item in weird_symbols:
    weird = cards_df_pop_parts[cards_df_pop_parts['part'].str.contains(item)]
    weird_cards.append(cards_df[cards_df['id'].isin(weird['card_id'])])
if weird_symbols:
    weird_cards = pd.concat(weird_cards)
    weird_cards[mains_col_names]

from itertools import chain
def get_increases(text_str, pat=r'([+-][\d+XxYx]{1,4}/[+-][\d+XxYx]{1,4})'):
    '''Given a text, extract a pattern and return the extraction or None'''
    res = re.findall(pat, text_str)
    return res

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
t = cards_df_pop_parts['part'].apply(get_increases)
pr_increase_symbols = set(chain(*(t.values)))
#pr_increase_symbols

# # Spacy applied

from spacy.symbols import ORTH, POS, NOUN, VERB, LOWER,LEMMA, TAG, nn#, VerbForm_inf,NounType_com,
import spacy
from spacy import displacy

# ## Load model

#MODEL = 'en_core_web_lg'
MODEL = 'en_core_web_sm'

# +
from spacy.tokens import Token

def get_token_sent(token):
    token_span = token.doc[token.i:token.i+1]
    return token_span.sent

try:
    Token.set_extension('sent', getter=get_token_sent, force=True)
except Exception:
    Token.set_extension('sent', getter=get_token_sent)
# -

#MODEL = r'C:\Users\cs294662\Downloads\programas\spacy\data\en_core_web_md-2.0.0\en_core_web_md\en_core_web_md-2.0.0'
#MODEL = r'C:\Users\cs294662\Downloads\programas\spacy\data\en_coref_lg-3.0.0\en_coref_lg\en_coref_lg-3.0.0'
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
nlp = spacy.load(MODEL)

# ### Learning

# + deletable=false editable=false run_control={"frozen": true}
# a=nlp('a')
# b=nlp('a')
# a==b
# -

# ## Set custom tags for special cases

symbols_set_valid = symbols_set.difference(set(worth_ignoring))
#symbols_set_valid
#symbols_explanation

# +
# Add {SYMBOL} to NOUN recognizer
symbols_set_mana = set()
symbols_set_action = set()
for sym in symbols_set_valid:
    if not sym in ['{T}', '{Q}', 'mana']:
        symbols_set_mana.add(sym)
        #nlp.tokenizer.add_special_case(sym, [{ORTH: sym, POS: NOUN, TAG:nn}])
        nlp.tokenizer.add_special_case(sym.upper(), [{ORTH: sym, POS: NOUN}])
        nlp.tokenizer.add_special_case(sym.lower(), [{ORTH: sym, POS: NOUN}])
    else:
        symbols_set_action.add(sym)
        nlp.tokenizer.add_special_case(sym, [{ORTH: sym, POS: VERB, TAG:'VB'}])

# Add power and toughness in/decresing symbols to NOUN recognizer
for sym in pr_increase_symbols:
    #nlp.tokenizer.add_special_case(sym, [{ORTH: sym, POS: NOUN, TAG:nn}])
    nlp.tokenizer.add_special_case(sym, [{ORTH: sym, POS: NOUN}])
# -

# ## Create custom entity matcher

# +
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span

class EntityPhraseMatcher(object):
    '''https://stackoverflow.com/questions/49097804/spacy-entity-from-phrasematcher-only'''
    
    name = 'entity_phrase_matcher'

    def __init__(self, nlp, terms, label):
        patterns = [nlp(term) for term in terms]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add(label, None, *patterns)

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []
        for label, start, end in matches:
            span = Span(doc, start, end, label=label)
            spans.append(span)
        doc.ents = spans
        return doc

def spans_overlaps(s1, s2):
    '''s1 and s2: spacy.tokens.span.Span
    If s1 and s2 overlaps, return True, else retrun False'''
    set1 = set(range(s1.start, s1.end+1))
    set2 = set(range(s2.start, s2.end+1))
    intersection = set1.intersection(set2)
    
    if intersection:
        return True
    return False

def get_longest_shortest_spans(s1, s2):
    '''s1 and s2: spacy.tokens.span.Span
    Returns s1, s2 if s1 longer or equal length to s2, else, return s2, s1'''
    l1 = s1.end - s1.start
    l2 = s2.end - s2.start
    
    if l1 >= l2:
        return s1, s2
    return s2, s1

    
class EntityMatcher(object):
    name = 'entity_matcher'

    def __init__(self, nlp, dict_label_terms):
        '''dict_label_terms shoould be a dictionary in the format
        {label(str): patterns(list)}'''
        self.matcher = Matcher(nlp.vocab)
        for label, patterns in dict_label_terms.items():
            self.matcher.add(label, None, *patterns)

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []
        for label, start, end in matches:
            span = Span(doc, start, end, label=label)
            spans.append(span)
            
            # One token can only have one entity (label)
            # So here, we need to check if this span intersects previous spans (and has different label, although two labels in overlaping spans is weird)
            # Then, take the longest span and discard the shortest one
            for old_span in spans:
                if spans_overlaps(span, old_span):
                    longest, shortest = get_longest_shortest_spans(span, old_span)
                    while shortest in spans: spans.remove(shortest)
                    while longest in spans: spans.remove(longest)  
                    spans.append(longest)
            
        doc.ents = spans
        return doc


# +
zones = ['graveyard', 'play', 'library', 'hand', 'battlefield', 'exile', 'stack']
players = ['opponent', 'you', 'controller', 'owner', 'player']
steps = ['upkeep', 'draw step', 'end step', 'cleanup step', 'main phase', 'main phases', 'combat damage step']
# actions = ['draw', 'discard'
#            'gain control', 'exchange control',
#            'deals', 'prevent',
#            # TODO gain life, gains
#            'destroy', 'counter', 'sacrifice',
#            'put'
#            ]

entities = {}
entities['zones'] = zones
entities['players'] = players
entities['steps'] = steps
entities['types'] = cards_types
entities['subtypes'] = cards_subtypes
entities['supertypes'] = cards_supertypes
# entities['actions'] = actions

# + code_folding=[]
# Create hashable dict
from collections import OrderedDict
import hashlib
class HashableDict(OrderedDict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
    
    def hexdigext(self):
        return hashlib.sha256(''.join([str(k)+str(v) for k, v in self.items()]).encode()).hexdigest()


# +
# Make defaultdict which depends on its key
# Source: https://www.reddit.com/r/Python/comments/27crqg/making_defaultdict_create_defaults_that_are_a/
from collections import defaultdict
class key_dependent_dict(defaultdict):
    def __init__(self, f_of_x):
        super().__init__(None) # base class doesn't get a factory
        self.f_of_x = f_of_x # save f(x)
    def __missing__(self, key): # called when a default needed
        ret = self.f_of_x(key) # calculate default value
        self[key] = ret # and install it in the dict
        return ret
    
def entity_key_hash(key):
    return HashableDict({'entity': key}).hexdigext()


# + code_folding=[]
# Create all entities
if 'ner' in nlp.pipe_names:
    nlp.remove_pipe('ner')
if 'entity_matcher' in nlp.pipe_names:
    nlp.remove_pipe('entity_matcher')
#nlp.remove_pipe('ent_type_matcher')
#nlp.remove_pipe('ent_subtype_matcher')
#nlp.remove_pipe('ent_supertype_matcher')

dict_label_terms = defaultdict(list)
entity_to_kind_map = {}
entity_key_to_hash_map = key_dependent_dict(entity_key_hash) # entity key: entity node hash (node_id)

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
for typ in cards_types:
    key = 'TYPE: ' + typ.lower()
    dict_label_terms[key].append([{'LOWER': t} for t in typ.lower().split()])
    dict_label_terms[key].append([{'LOWER': t+'s'} for t in typ.lower().split()])
    entity_to_kind_map[key] = 'TYPE'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
# TODO define plural for subtypes and types, like elves
cards_subtypes.add('elves')
for typ in cards_subtypes:
    key = 'SUBTYPE: ' + typ.lower()
    dict_label_terms[key].append([{'LOWER': t} for t in typ.lower().split()])
    dict_label_terms[key].append([{'LOWER': t+'s'} for t in typ.lower().split()])
    entity_to_kind_map[key] = 'SUBTYPE'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
for typ in cards_supertypes:
    key = 'SUPERTYPE: '+typ.lower()
    dict_label_terms[key].append([{'LOWER': t} for t in typ.lower().split()])
    dict_label_terms[key].append([{'LOWER': t+'s'} for t in typ.lower().split()])
    entity_to_kind_map[key] = 'SUPERTYPE'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
for typ in ['white','black','blue','white','red','green','colorless', 'multicolored', 'multicolor']:
    key = 'COLOR: '+typ.lower()
    dict_label_terms[key].append([{'LOWER': t} for t in typ.lower().split()])
    entity_to_kind_map[key] = 'COLOR'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
for abi in abilities:
    key = 'ABILITY: '+abi.lower()
    dict_label_terms[key].append([{'LOWER': t} for t in abi.lower().split()])
    entity_to_kind_map[key] = 'ABILITY'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
for zone in zones:
    key = 'ZONE: '+zone.lower()
    dict_label_terms[key].append([{'LOWER': t, 'POS': NOUN} for t in zone.lower().split()])
    dict_label_terms[key].append([{'LOWER': t+'s', 'POS': NOUN} for t in zone.lower().split()])
    entity_to_kind_map[key] = 'ZONE'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
for player in players:
    key = 'PLAYER: '+player.lower()
    dict_label_terms[key].append([{'LOWER': t, 'POS':spacy.symbols.PRON} for t in player.lower().split()])
    dict_label_terms[key].append([{'LOWER': t, 'POS':spacy.symbols.NOUN} for t in player.lower().split()])
    entity_to_kind_map[key] = 'PLAYER'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
for step in steps:
    key = 'STEP: '+step.lower()
    dict_label_terms[key].append([{'LOWER': t} for t in step.lower().split()])
    entity_to_kind_map[key] = 'STEP'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
for sym in symbols_set_mana:
    #print([{'ORTH': t} for t in sym.split()])
    key = 'MANA: '+sym.lower()
    if sym.strip('{}').isdigit() or sym.strip('{}').upper() == 'X':
        key = 'MANA: '+'{generic}'
    dict_label_terms[key].append([{'ORTH': t} for t in sym.split()])
    entity_to_kind_map[key] = 'MANA'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
for sym in symbols_set_action:
    key = 'ACTION: '+sym.lower()
    dict_label_terms[key].append([{'ORTH': t} for t in sym.split()])
    entity_to_kind_map[key] = 'ACTION'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
for sym in pr_increase_symbols:
    key = 'PT: '+sym.lower()
    dict_label_terms[key].append([{'ORTH': t} for t in sym.split()])
    entity_to_kind_map[key] = 'PT'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()
    
for sym in ['counter', 'card']:
    key = 'OBJECT: '+sym.lower()
    dict_label_terms[key].append([{'LOWER': t, 'POS': NOUN} for t in sym.split()])
    entity_to_kind_map[key] = 'OBJECT'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()

for sym in ['spell', 'hability', 'land']:
    key = 'NATURE: '+sym.lower()
    dict_label_terms[key].append([{'LOWER': t, 'POS': NOUN} for t in sym.split()])
    entity_to_kind_map[key] = 'NATURE'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()

for sym in ['self']:
    key = 'SELF: '+sym.lower()
    dict_label_terms[key].append([{'LOWER': t, 'POS': NOUN} for t in sym.split()])
    entity_to_kind_map[key] = 'SELF'
    entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()

# actions
key = 'VERBAL_ACTION: '
dict_label_terms[key].append([{'POS': VERB}])
entity_to_kind_map[key] = 'VERBAL_ACTION'
entity_key_to_hash_map[key] = HashableDict({'entity': key}).hexdigext()

entity_matcher = EntityMatcher(nlp, dict_label_terms)
try:
    nlp.add_pipe(entity_matcher, before='ner')
except Exception:
    nlp.add_pipe(entity_matcher)

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
logger.info(nlp.pipe_names)  # see all components in the pipeline


# -

# # Building graphs dfs

# ## Helping functions

# + code_folding=[]
# Function to parse docs to list of dicts
def parse_doc_to_list_of_dicts(df_row, original_cols=[], doc_col = 'part_doc'):
    '''Get a dataframe, parse the doc column to token and entity nodes and edges dict, return the dataframe'''
    doc = df_row[doc_col]
    
    token_node = [] # Source node
    token_node_text = []
    token_node_pos = []
    token_node_tag = []
    token_node_label = []
    
    token_head_dep = [] # Token edge do head
    token_head_node = [] # Target node
#     pop_node = [] # Pop node (avoided)
    part_node = [] # part node
    
    token_to_entity_edge = []
    entity_node = [] # Target entity nodes (target of token_node)
    entity_node_ent_type = []
    entity_node_entity = []
    entity_node_desc = []
    
    # Track relations between token_nodes
    tracker = HashableDict()
    
    # Avoided
#     pop_dic = HashableDict()
#     for col in pop_defining_cols:
#         pop_dic[col] = df_row[col]
    
    part_dic = HashableDict()
    for col in part_defining_cols:
        part_dic[col] = df_row[col]
    
    for t in doc:
        '''Create token and entity nodes and edges dict.'''
        
        token_dic = HashableDict()
        ent_dic = HashableDict()

        # Create node object as dict
        for col in ['card_id', 'paragraph_order', 'part_order', 'pop_order', 'part_type_full']:
            token_dic[col] = df_row[col]
        token_dic['text'] = t.text.lower()
        token_node_text.append(t.text.lower())
        #token_node_label.append(t.text.lower())
        token_dic['pos'] = t.pos_.lower()
        token_node_pos.append(t.pos_.lower())
        token_dic['tag'] = t.tag_.lower()
        token_node_tag.append(t.tag_.lower())
        token_dic['i'] = t.i
        
        # Add noun chunk and sents
        token_dic['noun_chunk_root'], token_dic['sent_root'] = 0, 0
        for np in doc.noun_chunks:
            if t in [s for s in np]:
                token_dic['noun_chunk'] = np.text
                if t == np.root:
                    token_dic['noun_chunk_root'] =1
        for sent in doc.sents:
            if t in [s for s in sent]:
                token_dic['sent'] = sent.text
                if t == sent.root:
                    token_dic['sent_root'] =1
        
        # Create entity node object as dict. All entities should be equal in all processed cards
        if t.ent_type_:
            
            ent = t.ent_type_
            ent_dic['entity'] = ent
            entity_node_entity.append(ent)
            typ, desc = ent.split(': ')
            entity_node_ent_type.append(typ)
            entity_node_desc.append(desc)
        
        else:
            entity_node_ent_type.append(pd.np.nan)
            entity_node_entity.append(pd.np.nan)
            entity_node_desc.append(pd.np.nan)
            
        token_node.append(token_dic.hexdigext())
        token_head_dep.append(t.dep_.lower())
        entity_node.append(ent_dic.hexdigext())
        token_to_entity_edge.append(t.ent_iob_.lower())
#         pop_node.append(pop_dic.hexdigext())
        part_node.append(part_dic.hexdigext())
        
        tracker[t] = {'token_dic': token_dic}
        
    # Now, set the head of a token as its target node
    for t, dicts in tracker.items():
        head = t.head
        head_dict = tracker[head]['token_dic']
        token_head_node.append(head_dict.hexdigext())
    
    # Create dataframe 
    res = pd.DataFrame()
    res['token_node'] = token_node
    res['token_node_text'] = token_node_text
    res['token_node_pos'] = token_node_pos
    res['token_node_tag'] = token_node_tag
    res['token_head_node'] = token_head_node
    res['token_head_dep'] = token_head_dep

    # Entity
    res['entity_node'] = entity_node
    res['entity_node_ent_type'] = entity_node_ent_type
    res['entity_node_entity'] = entity_node_entity
    res['entity_node_desc'] = entity_node_desc
    
    res['token_to_entity_edge'] = token_to_entity_edge
#     res['pop_node'] = pop_node # avoided
    res['part_node'] = part_node
    
    #res['label'] = token_node_label
    #res = res.reset_index(drop=True)

    for col in original_cols:
        res[col] = df_row[col]
    
    # If src and target are the same, the token is a root, set target to card_id
    res = res.reset_index(drop=True)
    try:
        if not res[res['token_head_dep'] == 'root'].index.empty:
            res.loc[res[res['token_head_dep'] == 'root'].index, 'token_head_node'] = res['part_node']
            res.loc[res[res['token_head_dep'] == 'root'].index, 'label'] = part_dic['part']
    except (TypeError) as e:
        # Someime res['token_head_dep'] = [] and cannot be compared to 'root'
        # ValueError: cannot set a frame with no defined index and a scalar
        pass
    
    return res


# + code_folding=[0]
# function to draw a graph to png
shapes = ['box', 'polygon', 'ellipse', 'oval', 'circle', 'egg', 'triangle', 'exagon', 'star']
colors = ['blue', 'black', 'red', '#db8625', 'green', 'gray', 'cyan', '#ed125b']
styles = ['filled', 'rounded', 'rounded, filled', 'dashed', 'dotted, bold']

entities_colors = {
    'PLAYER': '#FF6E6E',
    'ZONE': '#F5D300',
    'ACTION': '#1ADA00',
    'MANA': '#00DA84',
    'SUBTYPE': '#0DE5E5',
    'TYPE': '#0513F0',
    'SUPERTYPE': '#8D0BCA',
    'ABILITY': '#cc3300',
    'COLOR': '#666633',
    'STEP': '#E0E0F8'
}

def draw_graph(G, filename='test.png'):
    pdot = nx.drawing.nx_pydot.to_pydot(G)


    for i, node in enumerate(pdot.get_nodes()):
        attrs = node.get_attributes()
        node.set_label(str(attrs.get('label', 'none')))
    #     node.set_fontcolor(colors[random.randrange(len(colors))])
        entity_node_ent_type = attrs.get('entity_node_ent_type', pd.np.nan)
        if not pd.isnull(entity_node_ent_type):
            color = entities_colors[entity_node_ent_type.strip('"')]
            node.set_fillcolor(color)
            node.set_color(color)
            node.set_shape('hexagon')
            #node.set_colorscheme()
            node.set_style('filled')
        
        node_type = attrs.get('type', None)
        if node_type == '"card"':
            color = '#999966'
            node.set_fillcolor(color)
#             node.set_color(color)
            node.set_shape('star')
            #node.set_colorscheme()
            node.set_style('filled')
    #     
        #pass

    for i, edge in enumerate(pdot.get_edges()):
        att = edge.get_attributes()
        att = att.get('label', 'NO-LABEL')
        edge.set_label(att)
    #     edge.set_fontcolor(colors[random.randrange(len(colors))])
    #     edge.set_style(styles[random.randrange(len(styles))])
    #     edge.set_color(colors[random.randrange(len(colors))])

    png_path = filename
    pdot.write_png(png_path)

    from IPython.display import Image
    return Image(png_path)
# -

# ## Build the cards -> entities graphs (outgoing)
# REQUIRES TESTING
# Objective: card_id -> part -> root -> (children) -> entities

pop_defining_cols = ['card_id', 'paragraph_order', 'pop_order', 'pop_type', 'pop']
part_defining_cols = ['card_id', 'paragraph_order', 'pop_order', 'part_order', 'part_type_full', 'pop', 'part']

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
# Work on full cards_df_pop_parts (cards_df was probably filtered right at the beginning)
unique_card_ids = cards_df_pop_parts['card_id'].unique()
chunksize = 50
import datetime
start = datetime.datetime.now()
logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
for i in tqdm(range(unique_card_ids.shape[0]//chunksize)):
    
    if not ((i+1)*chunksize) % (10*chunksize):
        clear_output()
    logger.info('Working on {0} from {1}'.format((i+1)*chunksize, unique_card_ids.shape[0]))
    
    this_batch_ids = unique_card_ids[i*chunksize:(i+1)*chunksize]
    cards_df_for_graph = (cards_df_pop_parts[cards_df_pop_parts['card_id'].isin(this_batch_ids)]
                          .copy()
                         )
    cards_df_for_graph.loc[:, 'part_doc'] = cards_df_for_graph['part'].apply(lambda x: nlp(x.strip('.,')))

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # Parse for graph
    cards_df_for_graph = cards_df_for_graph.apply(parse_doc_to_list_of_dicts, args=(cards_df_pop_parts.columns,), axis=1)
    cards_df_for_graph = pd.concat(cards_df_for_graph.values, sort=False).reset_index(drop=True)
    #cards_df_for_graph.describe().transpose()
    
    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # Join with cards_df to get a few more details
    cdfg=cards_df_for_graph
    cdfg = cards_df_for_graph.merge(cards_df, left_on=['card_id'], right_index=True)

    # Try different approach: build node by node

    # Objective: card_id <- part <- root <- (children) <- entities # pop was avoided

    # Token to head
    # Manipulate df to generate two others: nodes and edges
    # Edges relate (token to head, head to part, part to pop, pop to card, and token to entity)
    # attention: head is also a token
    # and set node and edge attributes (both dfs should contain the attributes)

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # NODES ##########################################
    nodes = {}
    nodes_cols = {}
    nodes_attr = {}

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # Token nodes
    nodes_cols['token'] = ['token_node', 'token_node_text', 'token_node_pos', 'token_node_tag',
           'token_head_node', 'token_head_dep',
           'part_order', 'part_type', 'card_id', 'paragraph_order', 
           'pop_order', 'pop_type']
    nodes['token'] = (cdfg[nodes_cols['token']]
                      .rename(columns={'token_node':'node_id'})
                      .dropna(subset=['node_id'])
                     )
    nodes['token']['type'] = 'token'
    nodes['token']['label'] = nodes['token'].apply(lambda x:
                                                  '-'.join([x['token_node_text'],
                                                            x['token_node_pos'],
                                                            x['token_node_tag']]), axis=1)
    nodes_attr['token'] = [x for x in nodes['token'].columns if x not in ['node_id']]

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # Entity nodes
    nodes_cols['entity'] = ['entity_node', 'entity_node_entity','entity_node_ent_type', 'entity_node_desc']
    nodes['entity'] = (cdfg[nodes_cols['entity']]
                       .dropna(subset=['entity_node_ent_type'])
                       .rename(columns={'entity_node':'node_id'})
                      )
    nodes['entity']['type'] = 'entity'
    nodes['entity']['label'] = nodes['entity'].apply(lambda x:
                                                  '-'.join([x['entity_node_entity'],
                                                            ]), axis=1)
    nodes_attr['entity'] = [x for x in nodes['entity'].columns if x not in ['node_id']]

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # Part nodes
    nodes_cols['part'] = ['part_node',
                          'part', 'part_order', 'part_type',
                          'card_id',
                          'paragraph_order',
                          'pop_order', 'pop_type']
    nodes['part'] = (cdfg[nodes_cols['part']]
                     .rename(columns={'part_node':'node_id'})
                     .dropna(subset=['node_id'])
                     )
    nodes['part']['type'] = 'part'
    nodes['part']['label'] = nodes['part'].apply(lambda x:
                                                  '-'.join([x['part']]), axis=1)
    nodes_attr['part'] = [x for x in nodes['part'].columns if x not in ['node_id']]

    # Pop nodes (avoided)
    # nodes_cols['pop'] = ['pop_node',
    #                       'card_id',
    #                       'paragraph_order',
    #                       'pop', 'pop_order', 'pop_type']
    # nodes['pop'] = (cdfg[nodes_cols['pop']]
    #                 .rename(columns={'pop_node':'node_id'})
    #                 .dropna(subset=['node_id'])
    #                  )
    # nodes['pop']['type'] = 'pop'
    # nodes['pop']['label'] = nodes['pop'].apply(lambda x:
    #                                               '-'.join([x['pop']]), axis=1)
    # nodes_attr['pop'] = [x for x in nodes['pop'].columns if x not in ['node_id']]

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # Card nodes
    nodes_cols['card'] =  ['card_id'] + mains_col_names
    nodes['card'] = cdfg[nodes_cols['card']]
    nodes['card'] = nodes['card'].rename(columns={'name':'card_name'})
    nodes['card']['node_id'] = nodes['card']['card_id']
    nodes['card'] = nodes['card'].dropna(subset=['node_id', 'card_name'], how='any')                 
    nodes['card']['type'] = 'card'
    nodes['card']['label'] = nodes['card'].apply(lambda x:
                                                  '-'.join([x['card_name']]), axis=1)
    nodes_attr['card'] = [x for x in nodes['card'].columns if x not in ['node_id']]

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # EDGES #########################################
    card_as_start = True # Sets card as source and pop, part, token, entity as targets
    edges = {} # k->type, v-> dataframe
    edges_cols = {} # list
    edges_attr = {} # list

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # Token edges to head (and head to part)
    edges_cols['token_to_head_part'] = ['token_node', 'token_head_node', 'token_head_dep',
                           'part_order', 'part_type', 'card_id', 'paragraph_order', 
                           'pop_order', 'pop_type']
    renamer = {'token_node':'source', 'token_head_node':'target'}
    if card_as_start:
        renamer = {'token_head_node':'source', 'token_node':'target'}
    edges['token_to_head_part'] = (cdfg[edges_cols['token_to_head_part']]
                                   .rename(columns=renamer)
                                   .dropna(subset=['source', 'target'], how='any')
                                  )
    edges['token_to_head_part']['type'] = 'token_to_head_part'
    edges['token_to_head_part']['label'] = edges['token_to_head_part'].apply(lambda x:
                                                  '-'.join([x['token_head_dep'],
                                                           ]).upper(), axis=1)
    edges_attr['token_to_head_part'] = [x for x in edges['token_to_head_part'].columns
                                        if x not in ['source', 'target']]

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # Entity edges to Token
    edges_cols['entity_to_token'] = ['token_node', 'entity_node']
    renamer = {'entity_node':'source', 'token_node':'target'}
    if card_as_start:
        renamer = {'token_node':'source', 'entity_node':'target'}
    edges['entity_to_token'] = (cdfg[edges_cols['entity_to_token']]
                                .dropna()
                                .rename(columns=renamer)
                               )
    edges['entity_to_token']['type'] = 'entity_to_token'
    edges['entity_to_token']['relation'] = 'is_class_of'
    edges['entity_to_token']['label'] = edges['entity_to_token'].apply(lambda x:
                                                  '-'.join([x['relation'],
                                                           ]).upper(), axis=1)
    edges_attr['entity_to_token'] = [x for x in edges['entity_to_token'].columns
                                        if x not in ['source', 'target']]

    # Entity edges to cards
#     edges_cols['entity_to_card'] = ['card_id', 'entity_node']
#     renamer = {'entity_node':'source', 'card_id':'target'}
#     if card_as_start:
#         pass # Ignore in this case, because we want it reversed: card as target
#     #     renamer = {'card_id':'source', 'entity_node':'target'}
#     edges['entity_to_card'] = (cdfg[cdfg['edge_type']=='entity_to_card'][edges_cols['entity_to_card']]
#                                 .dropna()
#                                 .rename(columns=renamer)
#                                )
#     edges['entity_to_card']['type'] = 'entity_to_card'
#     edges['entity_to_card']['relation'] = 'is_contained_in'
#     edges['entity_to_card']['label'] = edges['entity_to_card'].apply(lambda x:
#                                                   '-'.join([x['relation'],
#                                                            ]).upper(), axis=1)
#     edges_attr['entity_to_card'] = [x for x in edges['entity_to_card'].columns
#                                         if x not in ['source', 'target']]

    # Part and pop edges (avoided)
    # edges_cols['part_to_pop'] = ['part_node', 'pop_node',
    #                        'part_order', 'part_type',
    #                        'card_id', 'paragraph_order', 
    #                        'pop_order', 'pop_type']
    # renamer = {'part_node':'source', 'pop_node':'target'}
    # if card_as_start:
    #     renamer = {'pop_node':'source', 'part_node':'target'}
    # edges['part_to_pop'] = (cdfg[edges_cols['part_to_pop']]
    #                         .rename(columns=renamer)
    #                         .dropna(subset=['source', 'target'], how='any')
    #                         )
    # edges['part_to_pop']['type'] = 'part_to_pop'
    # edges['part_to_pop']['label'] = edges['part_to_pop'].apply(lambda x:
    #                                               '-'.join([str(x['part_order']),
    #                                                         x['part_type'],
    #                                                        ]).upper(), axis=1)
    # edges_attr['part_to_pop'] = [x for x in edges['part_to_pop'].columns
    #                                     if x not in ['source', 'target']]

    # Pop to card edges (avoided)
    # edges_cols['pop_to_card'] = ['card_id', 'pop_node',
    #                        'paragraph_order', 
    #                        'pop_order', 'pop_type']
    # renamer = {'pop_node':'source', 'card_id':'target'}
    # if card_as_start:
    #     renamer = {'card_id':'source', 'pop_node':'target'}
    # edges['pop_to_card'] = (cdfg[edges_cols['pop_to_card']]
    #                         .rename(columns=renamer)
    #                         .dropna(subset=['source', 'target'], how='any')
    #                         )
    # edges['pop_to_card']['type'] = 'pop_to_card'
    # edges['pop_to_card']['label'] = edges['pop_to_card'].apply(lambda x:
    #                                               '-'.join([str(x['paragraph_order']),
    #                                                         str(x['pop_order']),
    #                                                         x['pop_type'],
    #                                                        ]).upper(), axis=1)
    # edges_attr['pop_to_card'] = [x for x in edges['pop_to_card'].columns
    #                                     if x not in ['source', 'target']]

    # Part and card edges (avoided)
    edges_cols['part_to_card'] = ['part_node', 'card_id',
                           'part_order', 'part_type_full',
                           'paragraph_order', 
                           'pop_order', 'pop_type', 'part_type']
    renamer = {'pop_node':'source', 'card_id':'target'}
    if card_as_start:
        renamer = {'card_id':'source', 'part_node':'target'}
    edges['part_to_card'] = (cdfg[edges_cols['part_to_card']]
                            .rename(columns=renamer)
                            .dropna(subset=['source', 'target'], how='any')
                            )
    edges['part_to_card']['type'] = 'part_to_card'
    edges['part_to_card']['label'] = edges['part_to_card'].apply(lambda x:
                                                  '-'.join([str(int(x['paragraph_order'])),
                                                            str(int(x['pop_order'])),
                                                            str(int(x['part_order'])),
                                                            x['part_type_full'],
                                                           ]).upper(), axis=1)
    edges_attr['part_to_card'] = [x for x in edges['part_to_card'].columns
                                        if x not in ['source', 'target']]

    # logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
    # Build dfs
    nodes_df = pd.concat(nodes.values(), sort=True).drop_duplicates(subset=['node_id'])
    nodes_df = nodes_df.dropna(subset=[x for x in nodes_df.columns if not x in ['node_id', 'label']], how='all')
    edges_df = pd.concat(edges.values(), sort=True).drop_duplicates(subset=['source', 'target'])
    edges_df = edges_df[
        (edges_df['source'].isin(nodes_df['node_id']))&
        (edges_df['target'].isin(nodes_df['node_id']))
    ]

    # Export
    #print('Exporting to pickle files')
    #nodes_df.to_pickle('./pickles/cards_outgoing_nodes_{0}.pkl'.format(i))
    #edges_df.to_pickle('./pickles/cards_outgoing_edges_{0}.pkl'.format(i))
    
    logger.info('Exporting to Postgres')
    method = 'append' if i else 'replace'
    nodes_df.set_index(['card_id', 'card_name', 'node_id', 'type']).to_sql(out_nodes_table_name, engine, if_exists=method)
    edges_df.set_index(['source', 'target', 'card_id', 'type']).to_sql(out_edges_table_name, engine, if_exists=method)
    
    time_decurred = datetime.datetime.now() - start
    logger.info('{0}'.format(time_decurred))

# ### Load individual pickles, join them and save an entire one

# + deletable=false editable=false run_control={"frozen": true}
# import os

# + deletable=false editable=false run_control={"frozen": true}
# dfs_nodes = []
# dfs_edges = []
# for path, dirname, filenames in os.walk('./pickles/'):
# #     print(path, dirname, filenames)
#     for i, filename in enumerate(filenames):
#         print('Processing {0}/{1} pickle files'.format(i,len(filenames)))
#         if 'cards_outgoing_edges_' in filename:
#             dfs_edges.append(pd.read_pickle(os.path.join(path, filename)))
#         if 'cards_outgoing_nodes_' in filename:
#             dfs_nodes.append(pd.read_pickle(os.path.join(path, filename)))
# nodes_df = pd.concat(dfs_nodes)
# edges_df = pd.concat(dfs_edges)
#
# print('Exporting to pickle files')
# nodes_df.to_pickle('./pickles/cards_outgoing_nodes.pkl')
# edges_df.to_pickle('./pickles/cards_outgoing_edges.pkl')
# print('Done exporting to pickle files')
# -

# ## Build the cards <- attributes graphs (incoming)
# Objective: card_id <- entities (attributes)

# cdfg=cards_df_for_graph
cdfg = cards_df#cards_df_for_graph.merge(cards_df, left_on=['card_id'], right_index=True)
cdfg['card_id']=cdfg.index
#cdfg

# +
# types|colors to cards graph
nodes_card_df = cdfg[['card_id', 'supertypes', 'types', 'subtypes', 'colors', 'manaCost']].copy()

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
# Generate df with card_id and entity_node_id refering to card's type, color, supertype, etc.
res = []
for col in nodes_card_df:
    if col == 'card_id':
        continue
    if col == 'manaCost':
        nodes_card_df[col] = nodes_card_df[col].apply(
            lambda x: ['{generic}'
                       if (y.strip('{}').isdigit() or y.strip('{}').upper()=='X')
                       else y
                       for y in re.findall(r'{.*?}', x)
                      ]
            if not pd.isnull(x) else x)
        nodes_card_df = nodes_card_df.rename(columns={'manaCost':'manas'})
        col = 'manas'
    temp = nodes_card_df[['card_id', col]].copy().dropna()
    temp = splitDataFrameList(temp, col)
    temp['entity_node_ent_type'] = col.upper()[:-1]
    # Build the name which can be maped to hexdigext
    temp['entity_node_entity'] = temp.apply(lambda x: ': '.join([x['entity_node_ent_type'], x[col].lower()]), axis=1)
    temp['entity_node'] = temp['entity_node_entity'].apply(lambda x: entity_key_to_hash_map[x])
    temp = temp.rename(columns={col: 'entity_node_desc'})
    temp = temp.drop_duplicates(subset=['card_id', 'entity_node'])
    res.append(temp)
    
res = pd.concat(res, sort=True)
res['edge_type'] = 'entity_to_card'

res.sample(5)
# -

logger.info(linecache.getline(__file__, inspect.getlineno(inspect.currentframe()) + 1))
cdfg = pd.concat([res, cdfg], sort=False).copy()
# cdfg['name'] = cdfg['card_name']

# +
# Try different approach: build node by node

# Objective: card_id <- part <- root <- (children) <- entities # pop was avoided

# Token to head
# Manipulate df to generate two others: nodes and edges
# Edges relate (token to head, head to part, part to pop, pop to card, and token to entity)
# attention: head is also a token
# and set node and edge attributes (both dfs should contain the attributes)

# NODES ########################################## 
nodes = {}
nodes_cols = {}
nodes_attr = {}


# Entity nodes
nodes_cols['entity'] = ['entity_node', 'entity_node_entity','entity_node_ent_type', 'entity_node_desc']
nodes['entity'] = (cdfg[nodes_cols['entity']]
                   .dropna(subset=['entity_node_ent_type'])
                   .rename(columns={'entity_node':'node_id'})
                  )
nodes['entity']['type'] = 'entity'
nodes['entity']['label'] = nodes['entity'].apply(lambda x:
                                              '-'.join([x['entity_node_entity'],
                                                        ]), axis=1)
nodes_attr['entity'] = [x for x in nodes['entity'].columns if x not in ['node_id']]


# Card nodes
nodes_cols['card'] =  ['card_id'] + mains_col_names
nodes['card'] = cdfg[nodes_cols['card']]
nodes['card'] = nodes['card'].rename(columns={'name':'card_name'})
nodes['card']['node_id'] = nodes['card']['card_id']
nodes['card'] = nodes['card'].dropna(subset=['node_id', 'card_name'], how='any')                 
nodes['card']['type'] = 'card'
nodes['card']['label'] = nodes['card'].apply(lambda x:
                                              '-'.join([x['card_name']]), axis=1)
nodes_attr['card'] = [x for x in nodes['card'].columns if x not in ['node_id']]

# EDGES #########################################
card_as_start = True # Sets card as source and pop, part, token, entity as targets
edges = {} # k->type, v-> dataframe
edges_cols = {} # list
edges_attr = {} # list

# Entity edges to cards
edges_cols['entity_to_card'] = ['card_id', 'entity_node']
renamer = {'entity_node':'source', 'card_id':'target'}
if card_as_start:
    pass # Ignore in this case, because we want it reversed: card as target
#     renamer = {'card_id':'source', 'entity_node':'target'}
edges['entity_to_card'] = (cdfg[cdfg['edge_type']=='entity_to_card'][edges_cols['entity_to_card']]
                            .dropna()
                            .rename(columns=renamer)
                           )
edges['entity_to_card']['type'] = 'entity_to_card'
edges['entity_to_card']['relation'] = 'is_contained_in'
edges['entity_to_card']['label'] = edges['entity_to_card'].apply(lambda x:
                                              '-'.join([x['relation'],
                                                       ]).upper(), axis=1)
edges_attr['entity_to_card'] = [x for x in edges['entity_to_card'].columns
                                    if x not in ['source', 'target']]

# Build dfs
nodes_df = pd.concat(nodes.values(), sort=True).drop_duplicates(subset=['node_id'])
nodes_df = nodes_df.dropna(subset=[x for x in nodes_df.columns if not x in ['node_id', 'label']], how='all')
edges_df = pd.concat(edges.values(), sort=True).drop_duplicates(subset=['source', 'target'])
edges_df = edges_df[
    (edges_df['source'].isin(nodes_df['node_id']))&
    (edges_df['target'].isin(nodes_df['node_id']))
]

# Export
#print('Exporting to pickle files')
#nodes_df.to_pickle('./pickles/cards_incoming_nodes.pkl')
#edges_df.to_pickle('./pickles/cards_incoming_edges.pkl')

logger.info('Exporting to Postgres')
method = 'replace'# if i else 
nodes_df.set_index(['card_id', 'card_name', 'node_id', 'type']).to_sql(in_nodes_table_name, engine, if_exists=method)
edges_df.set_index(['source', 'target', 'relation', 'type']).to_sql(in_edges_table_name, engine, if_exists=method)

logger.info('Done exporting to Postgres')

# + [markdown] heading_collapsed=true
# # Build graph from the dfs (reference only)

# + [markdown] heading_collapsed=true hidden=true
# ## Build graph with Networkx

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# import networkx as nx

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# def eliminate_and_wrap_in_quotes(text):
#     return '"'+str(text).replace('"', '')+'"'

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# # Create nodes and edges from dataframe
# graphs = []
# # EDGES
# source = 'source'
# target = 'target'
# for k in edges_df['type'].unique():
#     #if not k=='token_to_head_part': continue
#     print(k)
#     
#     edge_attr = edges_attr[k]
#     graphs.append(
#         nx.from_pandas_edgelist(edges_df[edges_df['type']==k],
#                               source=source,
#                               target=target,
#                               edge_attr=edge_attr,
#                               create_using=nx.DiGraph())
#     )
#
# G = nx.compose_all(graphs)
#
# # NODES (set attributes)
# for k in nodes_df['type'].unique():
#     print(k)
#     node_col = 'node_id'
#     for node_attr in nodes_attr[k]: 
#         temp = nodes_df[[node_attr, node_col]]
#         temp = temp.dropna()
#         
#         # Eliminate and wrap in quotes
#         temp[node_attr] = temp[node_attr].apply(eliminate_and_wrap_in_quotes)
#         nx.set_node_attributes(G, pd.Series(temp[node_attr].values, index=temp[node_col].values).copy().to_dict(), name=node_attr)

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# # Export to image
# draw_graph(G, 'Gtest.png')

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# # Build paths between each pair of cards
# card_nodes = [x for x,y in G.nodes(data=True) if y['type']=='"card"']
# temp = []
# for i, s in enumerate(card_nodes):
#     if not i%10: print(i)
#     for j, e in enumerate(card_nodes):
#         if s is e:
#             continue
#         # All simple paths becomes huge as it find path through many cards
#         # Lets try to create a different graph with only the nodes and edges with start card id, and the last node
#         # Entities do not have an attribute card_id
#         start_card_nodes = [x for x,y in G.nodes(data=True) if y.get('card_id', None) == G.node[s]['card_id']]
#         entity_nodes = [x for x,y in G.nodes(data=True) if y.get('type', None) == '"entity"']
#         interesting_subgraph = G.subgraph(start_card_nodes+entity_nodes+[e])
#         #temp_name = str('./paths_between_pairs/interesting_subgraph_{0}.png'.format(i))
#         #draw_graph(interesting_subgraph, temp_name)
#         for k, path in enumerate(nx.all_simple_paths(interesting_subgraph, s, e)):
#             subgraph = G.subgraph(path)
#             temp.append(G.subgraph(path))
# print('Avoid saving {0} images'.format(len(temp)))
# for i, g in enumerate(temp):
#     #if not i%10: print('{0}/{1}'.format(i, len(temp)))
#     #temp_name = str('./paths_between_pairs/temp_{0}.png'.format(i))
#     #draw_graph(g, temp_name)
#     #display(draw_graph(g, temp_name))
#     pass

# + [markdown] hidden=true
# ### Simplify link between cards

# + [markdown] hidden=true
# Let's contratct the paths to a simple link containing the path itself as attribute, but a simple label describing it.
#
# ATTENTION: This approach does not seem too useful

# + code_folding=[0] deletable=false editable=false hidden=true run_control={"frozen": true}
# # Build card -> root -> other_card (not too useful)
# import copy
# start_nodes = [x for x,y in G.nodes(data=True) if y['type']=='"card"']
# end_nodes = [x for x,y in G.nodes(data=True) if y['type']=='"card"']
# H = nx.DiGraph()
# for s in start_nodes:
#     for e in end_nodes:
#         if s is e:
#             continue
#         for path in nx.all_simple_paths(G, s, e):
#             H.add_nodes_from([(path[0], G.nodes[path[0]])])
#             H.add_nodes_from([(path[-1], G.nodes[path[-1]])])
#             #H.add_node(G[path[0]])
#             #H.add_node(G[path[-1]])
#             # Build edge attributes and than add it (after the loop)
#             att = {}
#             label = ''
#             added_node = None
#             edge1_label = ''
#             edge2_label = ''
#             for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
#                 if not i:
#                     edge1_label += G.edges[a,b].get('label','')
#                 if G.nodes[a].get('token_head_dep','').strip('"') =='root': #build root node
#                     added_node = copy.deepcopy(a)
#                     added_node_attr = G.nodes[a]
#                     added_node_attr.update({'full_original_path':path})
#                     H.add_nodes_from([(added_node, added_node_attr)])
#                     edge2_label += G.nodes[b].get('token_head_dep','none').strip('"')
#             
#             H.add_edge(path[0], added_node, label=edge1_label)
#             H.add_edge(added_node, path[-1], label=edge2_label)
# H_temp_name = str('H.png'.format(i))
# display(draw_graph(H, H_temp_name))

# + [markdown] heading_collapsed=true hidden=true
# ### Get paths between cards and join them, contracting same card nodes

# + code_folding=[] deletable=false editable=false hidden=true hideCode=false run_control={"frozen": true}
# # Card to card paths (degree here may be interesting)
# # ATTENTION: TAKES A LONG TIME. SHOULD BE OPTIMIZED
# '''
# import copy
# H = nx.DiGraph()
# card_nodes = [x for x,y in G.nodes(data=True) if y['type']=='"card"']
# temp = []
# for i, s in enumerate(card_nodes):
#     if not i%10: print(i)
#     for j, e in enumerate(card_nodes):
#         if s is e:
#             continue
#             
#         # All simple paths becomes huge as it find path through many cards
#         # Lets try to create a different graph with only the nodes and edges with start card id, and the last node
#         # Entities do not have an attribute card_id
#         start_card_nodes = [x for x,y in G.nodes(data=True) if y.get('card_id', None) == G.node[s]['card_id']]
#         entity_nodes = [x for x,y in G.nodes(data=True) if y.get('type', None) == '"entity"']
#         interesting_subgraph = G.subgraph(start_card_nodes+entity_nodes+[e])
#         #temp_name = str('./paths_between_pairs/interesting_subgraph_{0}.png'.format(i))
#         #draw_graph(interesting_subgraph, temp_name)
#         for k, path in enumerate(nx.all_simple_paths(interesting_subgraph, s, e)):
#             subgraph = G.subgraph(path)
#             H = nx.union(H, subgraph, rename=('H-', 'path-'))
#
# # Contract all card nodes, so all edges begin and end at a card
# # Comment this chunk and you will get all disjoint paths between cards
# card_names = set([y['card_name'] for x,y in H.nodes(data=True) if y['type'].strip('"')=='card'])
# groups_of_same_nodes = []
# print("Start grouping")
# for i, card_name in enumerate(card_names):
#     if not i%10: print('{0}/{1}'.format(i, len(card_names)))
#     temp = [x for x,y in H.nodes(data=True) if y.get('card_name','')==card_name]
#     if len(temp)>1:
#         groups_of_same_nodes.append(temp)
# print("Start contraction")
# for i, group in enumerate(groups_of_same_nodes):
#     if not i%10: print('{0}/{1}'.format(i, len(groups_of_same_nodes)))
#     for node in group[1:]:
#         H = nx.contracted_nodes(H, group[0], node)
# '''

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# # THIS TAKES FOREVER
# # Print to file 
# print("Start writing image")
# H_temp_name = str('H.png'.format(i))
# display(draw_graph(H, H_temp_name))

# + [markdown] hidden=true
# ### Export

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# # Export to cytoscape format
# print('Export graphml G')
# nx.write_graphml(G, 'G_test.graphml')
# print('Export graphml H')
# # Remove attributes of type dict
# for (n,d) in H.nodes(data=True):
#     if d.get("contraction", None):
#         del d["contraction"]
# nx.write_graphml(H, 'H_test.graphml')

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# # To nodes and edges table in postgresql
# nodes_df['id'] = nodes_df['node_id']
# nodes_df.to_sql('nodes', engine, index=False, if_exists='replace')
# edges_df.to_sql('edges', engine, index=False, if_exists='replace')
# -

# # Compute metrics (reference only)

# ## Count edges and their type related to nodes which are entites instances

# + deletable=false editable=false run_control={"frozen": true}
# entity_nodes = nodes_df[nodes_df['type']=='entity']
# test_ent_id = entity_nodes.iloc[0]['node_id']
# print(test_ent_id)
# entity_nodes

# + deletable=false editable=false run_control={"frozen": true}
# in_edges = []
# out_edges = []
# res = []
# for ent_node in entity_nodes['node_id']:
#     for node in G[ent_node]: # node is a neighbour of the entity
#         for in_ed in G.in_edges([node], data=True):
#             s, t, d = in_ed #source, target, data
#             in_edges.append(d)
#         for out_ed in G.out_edges([node], data=True):
#             s, t, d = out_ed #source, target, data
#             out_edges.append(d)
#
#     in_ = pd.DataFrame(in_edges)
#     in_['edge_type'] = 'in'
#     in_['ent_node'] = nx.get_node_attributes(G, 'label')[ent_node]
#     out_ = pd.DataFrame(out_edges)
#     out_['edge_type'] = 'out'
#     out_['ent_node'] = nx.get_node_attributes(G, 'label')[ent_node]
#     res.append(pd.concat([in_, out_]).copy())
#     
# res = pd.concat(res)
# res['cont'] = 1
# res

# + deletable=false editable=false run_control={"frozen": true}
# res.pivot_table(values=['cont'], index=['ent_node', 'label', 'pop_type'], columns=['edge_type'], aggfunc=pd.np.sum)

logger.info(f'FINISHED: {__file__}')