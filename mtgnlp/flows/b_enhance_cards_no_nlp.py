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

import uuid
import collections
from sqlalchemy import create_engine
from tqdm import tqdm
import json
import pandas as pd
import numpy as numpy
import re
from collections import defaultdict

import logging
import inspect
import linecache
import os

from mtgnlp import config

try:
    __file__
except NameError:
    # for running in ipython
    fname = "01.01-cards-ETL-no-NLP.py"
    __file__ = os.path.abspath(os.path.realpath(fname))

logPathFileName = "./logs/" + "01.01.log"

# create logger'
logger = logging.getLogger("01.01")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{logPathFileName}", mode="w")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# This is for jupyter notebook
# from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()
# This is for terminal
logger.info("from tqdm import tqdm")

tqdm.pandas(desc="Progress")

# %% Load cards set

# + hideCode=false
allprintings_fname = config.ALL_PRINTINGS_PATH
sets = json.load(open(allprintings_fname, "rb"))["data"]

cards_all = []
for k, sett in sets.items():
    if (k in ["UGL", "UST", "UNH"]) or (
        len(k) > 3
    ):  # Ignore Unglued, Unstable and promotional things
        continue
    for card in sett["cards"]:
        card["set"] = k
        card["set_code"] = sets[k]["code"]
        card["set_releaseDate"] = sets[k]["releaseDate"]
        card["set_baseSetSize"] = sets[k]["baseSetSize"]
        card["set_totalSetSize"] = sets[k]["totalSetSize"]
        card["set_name"] = sets[k]["name"]
    cards_all.extend(sett["cards"])

# # Params

ASPAS_TEXT = "ASPAS_TEXT"

mains_col_names = [
    "name",
    "manaCost",
    "text_preworked",
    "type",
    "power",
    "toughness",
    "types",
    "supertypes",
    "subtypes",
]


engine = create_engine(config.DB_STR)
engine.connect()

logger.info(engine.connect())

export_table_name = "cards_text_parts"

# %% Helping functions

# + code_folding=[0]
# Split dataframelist


def splitDataFrameList(df, target_column, separator=None):
    """
    https://gist.github.com/jlln/338b4b0b55bd6984f883
    df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    """

    def splitListToRows(row, row_accumulator, target_column, separator):
        split_row = row[target_column]  # .split(separator)
        if isinstance(split_row, collections.Iterable):
            for s in split_row:
                new_row = row.to_dict()
                new_row[target_column] = s
                row_accumulator.append(new_row)
        else:
            new_row = row.to_dict()
            new_row[target_column] = numpy.nan
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(splitListToRows, axis=1, args=(new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


# -

# # Create dataframe of cards

# %% Create tables for cards

cards = cards_all
cards_df = pd.DataFrame.from_dict(cards)
cards_df = cards_df.drop_duplicates(subset=["name"])
cards_df = cards_df.drop(
    columns=[
        "foreignData",
        "legalities",
        "prices",
        "purchaseUrls",
        "rulings",
        "leadershipSkills",
    ],
    errors="ignore",
)
cards_df["set_releaseDate"] = pd.to_datetime(
    cards_df["set_releaseDate"], format="%Y-%m-%d"
)

# cards_df = cards_df.sample(200)
# cards_df = cards_df[cards_df['name'].isin(all_cards_names_in_decks)]

# ### Add and transform features
logger.info("p/t")
# + code_folding=[]
# Make numeric power and toughness
cards_df["power_num"] = pd.to_numeric(cards_df["power"], errors="coerce")
cards_df["toughness_num"] = pd.to_numeric(cards_df["toughness"], errors="coerce")
# -

logger.info("mana costs")
# Add colored and generic cmcs
# Find all patterns like {.*?} [1], than find all {\d} or {X} (this is generic) [2]
# Subtract the len([2]) from len([1])
all_mana_pattern = r"{.*?}"
generic_mana_pattern = r"{(?:X|\d)}"
cards_df["id"] = cards_df["uuid"]
assert not cards_df[["id"]].isnull().values.any()
cards_df["manaCost_tuples_generic"] = cards_df["manaCost"].apply(
    lambda x: list(re.findall(generic_mana_pattern, str(x)))
)
cards_df["manaCost_tuples"] = cards_df["manaCost"].apply(
    lambda x: list(re.findall(all_mana_pattern, str(x)))
)
cards_df["manaCost_tuples_len"] = cards_df["manaCost_tuples"].apply(lambda x: len(x))
cards_df["manaCost_tuples_generic_len"] = cards_df["manaCost_tuples_generic"].apply(
    lambda x: len(x)
)
cards_df["manaCost_coloured"] = (
    cards_df["manaCost_tuples_len"] - cards_df["manaCost_tuples_generic_len"]
)
cards_df["cmc"] = cards_df["convertedManaCost"]
cards_df["manaCost_generic"] = cards_df["cmc"] - cards_df["manaCost_coloured"]

# +
# %% Replace name by SELF and remove anything between parethesis
pattern_parenthesis = r" ?\(.*?\)"

logger.info("apply text prework")


def prework_text(card):
    t = str(card["text"]).replace(card["name"], "SELF")
    t = re.sub(pattern_parenthesis, "", t)
    return t


cards_df["text_preworked"] = cards_df.apply(prework_text, axis=1)


# cards_df['text_preworked']

# +
# %% Set land text, which may be empty
def add_mana_text(text, sym):
    if not text:
        return "{T}: Add " + sym + "."
    elif "{T}: Add " + sym not in text:
        return text + "\n" + "{T}: Add " + sym + "."
    return text


lands = [
    ("Plains", "{W}"),
    ("Swamp", "{B}"),
    ("Island", "{U}"),
    ("Mountain", "{R}"),
    ("Forest", "{G}"),
]
for land_name, sym in lands:
    cards_df["text_preworked"] = cards_df.progress_apply(
        lambda x: add_mana_text(x["text_preworked"], sym)
        if isinstance(x["subtypes"], list) and land_name in x["subtypes"]
        else x["text_preworked"],
        axis=1,
    )
# -
logger.info("has add")
# %% Check whether card can add mana
cards_df["has_add"] = cards_df["text_preworked"].apply(
    lambda x: True if re.findall(r"add ", str(x), flags=re.IGNORECASE) else False
)

logger.info("text preworked")

sep = "ª"
if cards_df["text_preworked"].str.contains(sep).any():
    raise Exception("Bad separator symbol. It is contained in some text.")

assert cards_df[cards_df["text_preworked"].str.contains("\(").fillna(False)][
    "text_preworked"
].empty

# %% Export

# + deletable=false editable=false run_control={"frozen": true}
# # Convert lists and dicts to json to avoid ProgrammingError: (psycopg2.ProgrammingError) can't adapt type 'dict'
# cards_df_export = .copy()
# types = {col:type(cards_df_export.iloc[0][col]) for col in cards_df_export.columns}
#
# for col, typ in types.items():
#     if type in ['list', 'dict']:
#         import json
#         cards_df_export[col] = cards_df_export[col].apply(lambda x: json.dumps(x))
# -

# Export to sql
logger.info("cards_df to sql")
cards_df.set_index(["id", "name"]).to_sql(
    config.CARDS_TNAME, engine, if_exists="replace"
)

# +
# Export keys tables
logger.info("unique_ids to sql")
unique_ids = pd.DataFrame(cards_df["id"].unique())
unique_ids.to_sql("unique_card_ids", engine, index=False, if_exists="replace")

logger.info("unique_names to sql")
unique_names = cards_df[["name"]].drop_duplicates()
unique_names.to_sql("unique_card_names", engine, index=False, if_exists="replace")

logger.info("unique_card_ids_names to sql")
unique_card_ids_names = cards_df[["id", "name"]].drop_duplicates(subset=["name"])
unique_card_ids_names.to_sql(
    "unique_card_ids_names", engine, index=False, if_exists="replace"
)
# -

# %% Create accessory tables form lists

cols_containing_lists = []
for col, ob in zip(cards_df.iloc[0].index, cards_df.iloc[0]):
    # print(i, ob, type(ob))
    if isinstance(ob, list):
        cols_containing_lists.append(col)
cols_containing_lists

# %% Export

for col in cols_containing_lists:
    logger.info(f"Procenssing col containing list: {col}")
    temp = cards_df[["name", col]].drop_duplicates(subset=["name"])
    temp = splitDataFrameList(temp, col)
    if col in ["colorIdentity", "colors"]:
        temp[col] = temp[col].fillna("colorless")
    temp = temp.dropna().set_index("name")
    temp.to_sql("cards_" + col, engine, if_exists="replace")

# %% Finishing

# Set id as index for later work
cards_df = cards_df.set_index("id")

# # Domain specific vocabulary

# Let's build some domain specific vocabulary for MTG. For example, let's list supertypes, types, subtypes, know all card names, this kind f thing.

# %% Create set of cards names
cards_names = set(cards_df.name.unique())

# +
# %% Create set of supertypes
array_of_supertypes_tuples = cards_df["supertypes"].dropna().apply(tuple).unique()
cards_supertypes = tuple()
for tup in array_of_supertypes_tuples:
    cards_supertypes += tup

cards_supertypes = set(cards_supertypes)
cards_supertypes

# +
# %% Create set of types
array_of_types_tuples = cards_df["types"].dropna().apply(tuple).unique()
cards_types = tuple()
for tup in array_of_types_tuples:
    cards_types += tup

cards_types = set(cards_types)
# cards_types

# +
# Create set of types
array_of_subtypes_tuples = cards_df["subtypes"].dropna().apply(tuple).unique()
cards_subtypes = tuple()
for tup in array_of_subtypes_tuples:
    cards_subtypes += tup

cards_subtypes = set(cards_subtypes)
# cards_subtypes

# +
# cards_df.head(10).transpose()

# + deletable=false editable=false run_control={"frozen": true}
# import requests
# import pickle
# r = requests.get('https://media.wizards.com/2020/downloads/MagicCompRules%2020200122.txt')
# if not r.status_code == 200:
#     r.raise_for_status()
# comprules = r.text
# -

# %% Process rules
with open("rules.txt", "r", encoding="latin-1") as f:
    comprules = "\n".join(f.readlines())

kw_abilities_pat = r"702\.\d+\. ([A-Za-z ]+)"
abilities = re.findall(kw_abilities_pat, comprules, re.IGNORECASE)
abilities.pop(0)  # Its just the rulings
abilities.sort()
# abilities

# %% Detect an abilities sentence?

# We should:
# - Split sentences in a card by '\n' (=card_sentences_list)
# - Split each element in card_sentences_list by ', ' (=split_candidate_sentences)
# - Search for the pattern r'^ability' in each item of split_candidate_sentences
# - If the pattern is found for evey item, then, split_candidate_sentences is an abilities sentence
#
# We can, at the same time, detect activated abilites sentences and "rest" sentences (which are not abilites and not triggered abilites ones).
# - Split sentences in a card by '\n' (=card_sentences_list)
# - Those sentences which contain : are activated abilites
#
# Sentences which are not in any case above are "rest" sentences.

# + code_folding=[]
ability_start_pattern = r"|".join(["^" + ab + r"\b" for ab in abilities])


# print(ability_start_pattern)
def is_ability_sentence(sentence):
    elem_starting_with_ability = []
    exceptions = ["Cycling abilities you activate cost up to {2} less to activate."]
    if sentence in exceptions:
        return False
    elems = sentence.replace(";", ",").split(", ")
    for elem in elems:
        if re.search(ability_start_pattern, elem, re.IGNORECASE):
            elem_starting_with_ability.append(
                re.search(ability_start_pattern, elem, re.IGNORECASE)
            )
        else:
            return False
    if len(elems) == len(elem_starting_with_ability):
        return True
    raise Exception("We should never get here")


# -

# %% Lets detetect all paragraphs types (and keep each ability as a separate paragraph)


# + code_folding=[1]


def splitDataFrameList(df, target_column, separator=None):
    """
    https://gist.github.com/jlln/338b4b0b55bd6984f883
    df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    """

    def splitListToRows(row, row_accumulator, target_column, separator):
        split_row = row[target_column]  # .split(separator)
        if isinstance(split_row, collections.Iterable):
            for s in split_row:
                new_row = row.to_dict()
                new_row[target_column] = s
                row_accumulator.append(new_row)
        else:
            new_row = row.to_dict()
            new_row[target_column] = numpy.nan
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(splitListToRows, axis=1, args=(new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


# + code_folding=[0, 8, 15, 28]
def get_paragraph_type(paragraph):
    if is_ability_sentence(paragraph):
        return "ability"
    elif ":" in paragraph:
        return "activated"
    else:
        return "rest"


def split_abilities_and_keep_the_rest(df_row):
    """Returns a list of abilities or a list of one element, which is not ability"""
    if df_row["paragraph_type"] == "ability":
        return [x.strip() for x in df_row["paragraph"].replace(";", ",").split(",")]

    return [df_row["paragraph"]]


def get_aspas(text):
    if pd.isnull(text):
        return numpy.nan

    reg = re.findall(r"\"(.+?)\"", text)

    if not reg:
        return numpy.nan

    res = reg[0]

    return res


def get_paragraphs_and_types_df(card_row):
    res = pd.DataFrame()
    temp = pd.DataFrame()

    # Get initial paragraphs
    temp["paragraph"] = card_row["text_preworked"].split("\n")
    temp[ASPAS_TEXT] = temp["paragraph"].apply(get_aspas)
    temp["paragraph"] = temp.apply(
        lambda x: x["paragraph"].replace(x[ASPAS_TEXT], ASPAS_TEXT)
        if not pd.isnull(x[ASPAS_TEXT])
        else x["paragraph"],
        axis=1,
    )

    temp["paragraph_type"] = temp["paragraph"].apply(get_paragraph_type)

    # Split the abilities paragraphs into multiple rows
    temp["paragraph"] = temp.apply(split_abilities_and_keep_the_rest, axis=1)
    temp = splitDataFrameList(temp, "paragraph")
    res = temp

    res["card_id"] = card_row.name
    res["paragraph_order"] = range(res.shape[0])
    return res


# -
logger.info(
    """cards_df['df_paragraphs'] = cards_df.progress_apply(get_paragraphs_and_types_df, axis=1)"""
)
cards_df["df_paragraphs"] = cards_df.progress_apply(get_paragraphs_and_types_df, axis=1)

# +
# cards_df[['text_preworked','df_paragraphs']].iloc[21]['df_paragraphs']
# -

cards_df_paragraphs = pd.concat(cards_df["df_paragraphs"].values)
cards_df_paragraphs.head(4)

# + deletable=false editable=false run_control={"frozen": true}
# # Show cards with triggered abilities
# cards_df[cards_df['df_sentences'].apply(lambda x: 'activated' in x['type'].values)]
# -

# %% Lets use the same approach and separate paragraphs in abilities-complements, costs-effects and keep the rest as is

ability_and_complement_regex = r"(" + ability_start_pattern + ")" + r"(.*)"


# ability_and_complement_regex


def get_pop_and_complements_df(paragraph_row):
    res = pd.DataFrame()
    pat_ability = re.compile(ability_and_complement_regex, re.IGNORECASE)

    if paragraph_row["paragraph_type"] == "ability":

        # print(res['pop'].iloc[0])
        # print(re.findall(pat, res['pop'].iloc[0]))
        x = paragraph_row["paragraph"]
        if (not pd.isnull(x)) and re.findall(pat_ability, x):
            ability = re.findall(pat_ability, x)[0][0].strip()
            ability_complement = re.findall(pat_ability, x)[0][1].strip()
        else:
            import pdb

            pdb.set_trace()

        res["pop"] = [ability, ability_complement]
        res["pop_type"] = ["ability", "ability_complement"]
        res["pop_order"] = range(res["pop"].shape[0])

    elif paragraph_row["paragraph_type"] == "activated":
        """Break the costs in individual ones"""
        costs, effect = paragraph_row["paragraph"].split(":")

        exceptions = ["Pay half your life, rounded up"]
        if costs in exceptions:
            costs = costs.replace(",", "")

        res["pop"] = costs.split(",") + [effect]
        types = ["activation_cost" for x in costs.split(",")] + ["activated_effect"]

        res["pop_type"] = types
        res["pop_order"] = range(res["pop"].shape[0])

    else:
        """Keep the rest as rest or effect"""
        effect = paragraph_row["paragraph"]

        res["pop"] = [effect]
        res["pop_type"] = ["effect"]
        res["pop_order"] = range(res["pop"].shape[0])

    res["card_id"] = paragraph_row["card_id"]
    res["paragraph_order"] = paragraph_row["paragraph_order"]
    res["paragraph_type"] = paragraph_row["paragraph_type"]
    res["paragraph"] = paragraph_row["paragraph"]
    return res


logger.info(
    """cards_df_paragraphs['pop'] = cards_df_paragraphs.progress_apply(get_pop_and_complements_df, axis=1)"""
)
cards_df_paragraphs["pop"] = cards_df_paragraphs.progress_apply(
    get_pop_and_complements_df, axis=1
)

# + deletable=false editable=false run_control={"frozen": true}
# cards_df_paragraphs.iloc[3]['pop']
# -
logger.info(
    """cards_df_pops = pd.concat(cards_df_paragraphs['pop'].values, sort=True)"""
)
cards_df_pops = pd.concat(cards_df_paragraphs["pop"].values, sort=True)
# cards_df_pops['pop_hash'] = cards_df_pops['pop'].apply(lambda x: uuid.uuid4().hex)
# cards_df_pops.sort_values(by=['card_id','paragraph_order','pop_order']).head(3)

# + deletable=false editable=false run_control={"frozen": true}
# cards_df_pops[cards_df_pops['pop_type']=='activation_cost']['pop'].dropna().unique()

# + deletable=false editable=false run_control={"frozen": true}
# # Count how many abilities, activated abilities and effects there are
# temp = cards_df_pops
# temp['cont'] = 1
#
# index = ['pop_type']
# values = ['cont']
#
# pivot_pop = temp.pivot_table(index=index, values=values, aggfunc=numpy.sum)
# pivot_pop
# -

# %% Lets use the same approach and separate conditions-"result effect"

condition_regex = r"((?:if |whenever |when |only |unless |as long as ).*?[,.])"
# condition_regex

intensifier_regex = r"((?:for each ).*?[,.])"
# intensifier_regex

step_condition_regex = r"(at the (?:beginning |end )of.*?[,.])"


# step_condition_regex

# + code_folding=[0]
def get_conditions_and_effects_df(pop_row, original_cols=[]):
    res = pd.DataFrame()
    text = pop_row["pop"]

    # Get list of conditions in text
    reg_cond = re.findall(condition_regex, text, flags=re.IGNORECASE)
    if not reg_cond:
        reg_cond = []

    # Get list of step (time) conditions in text
    reg_step_cond = re.findall(step_condition_regex, text, flags=re.IGNORECASE)
    if not reg_step_cond:
        reg_step_cond = []

    # Get list of intensifiers in text
    reg_intensifier = re.findall(intensifier_regex, text, flags=re.IGNORECASE)
    if not reg_intensifier:
        reg_intensifier = []

        # Get the rest of the text in a list
    text_wo_conditions = text
    for cond in reg_cond + reg_step_cond + reg_intensifier:
        text_wo_conditions = text_wo_conditions.replace(cond, "")
    text_wo_conditions = text_wo_conditions.strip(",. ")
    text_wo_conditions = [text_wo_conditions]

    temp = []
    for part in reg_cond:
        temp.append(
            {
                "part_order": text.find(part),
                "part": part.strip(",. "),
                "part_type": "condition",
            }
        )
    for part in reg_step_cond:
        temp.append(
            {
                "part_order": text.find(part),
                "part": part.strip(",. "),
                "part_type": "step_condition",
            }
        )
    for part in reg_intensifier:
        temp.append(
            {
                "part_order": text.find(part),
                "part": part.strip(",. "),
                "part_type": "intensifier_for_each",
            }
        )
    for part in text_wo_conditions:
        temp.append(
            {
                "part_order": text.find(part),
                "part": part.strip(",. "),
                "part_type": "wo_conditions",
            }
        )

    # Reset order to start from zero
    res = pd.DataFrame(temp).sort_values(by=["part_order"])
    res = res.reset_index(drop=True)
    res["part_order"] = res.index

    for col in original_cols:
        res[col] = pop_row[col]

    return res


logger.info("cards_df_pops['pop_part'] = cards_df_pops.progress_apply")
cards_df_pops["pop_parts"] = cards_df_pops.progress_apply(
    get_conditions_and_effects_df, args=(cards_df_pops.columns,), axis=1
)
cards_df_pop_parts = pd.concat(cards_df_pops["pop_parts"].values)

# + code_folding=[]
# cards_df_pop_parts
# -

# %% Detect named cards cited inside cards text

# For later: define a way to get card named cited in other cards text. Same approach of self should suffice:
# 1. Detect the names (done below)
# 2. Replace the names with a place holder. CARD_NAME_1, CARD_NAME_2 (for each card name in a cards text).
# 3. Create columns CARD_NAME_1, CARD_NAME_2, etc. in dataframe, holding the actual name in the cell value
# 4. Create entity detector for CARD_NAME_1, CARD_NAME_2,...
# 5. Manually add edge between CARD_NAME_1 and its actual value (the actual card name)

named_card_pattern = r"(" + r"|".join(["{0}".format(n) for n in cards_names]) + r")"
named_card_regex = (
    r" named " + named_card_pattern + "((?: or )" + named_card_pattern + ")?" + r".*?"
)
# named_card_regex

# + [markdown] heading_collapsed=true
# ### Tests

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# test_text = 'Add {G} for every card named Path of Peace in all graveyards.'
# test = re.findall(named_card_regex, test_text)
# test

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# a = cards_df['text_preworked'].apply(
#     lambda x: re.findall(named_card_regex, x)
#     if re.findall(named_card_regex, x)
#     else numpy.nan
# ).dropna()

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# 'Zhang Fei, Fierce Warrior' in named_card_regex

# + deletable=false editable=false hidden=true hideCode=false run_control={"frozen": true}
# cards_df.loc[a.index[0]]['text_preworked']

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# a.iloc[0]

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# test_text = 'SELF gets +2/+2 as long as you control a permanent named Guan Yu, Sainted Warrior or a permanent named Zhang Fei, Fierce Warrior in the battlefield.'
# test = re.findall(named_card_regex, test_text)
# test

# + deletable=false editable=false hidden=true hideCode=false run_control={"frozen": true}
# cards_df.loc['ef0fe275d7e5625b20f4c5cd7fc34301df0bea6d']['text_preworked']

# + deletable=false editable=false hidden=true run_control={"frozen": true}
# a['ef0fe275d7e5625b20f4c5cd7fc34301df0bea6d']
# -

# %% reate key

# + deletable=false editable=false run_control={"frozen": true}
# #cards_df_pop_parts = pd.read_sql_table('cards_text_parts', engine)
# -
logger.info("cards_df_pop_parts['text_pk'] = cards_df_pop_parts.progress_apply")
cards_df_pop_parts["text_pk"] = cards_df_pop_parts.progress_apply(
    lambda x: "-".join(
        [
            x["card_id"],
            str(x["paragraph_order"]),
            str(x["pop_order"]),
            str(x["part_order"]),
        ]
    ),
    axis=1,
)

# %% Drop empty pops
logger.info(
    "cards_df_pop_parts['part'] = cards_df_pop_parts['part'].replace(" ", numpy.nan)"
)
cards_df_pop_parts["part"] = cards_df_pop_parts["part"].replace("", numpy.nan)
logger.info("Logging to get line number")
cards_df_pop_parts = cards_df_pop_parts.dropna(subset=["part"])

# %% Avoid pop

# + code_folding=[]
# Lets avoid creating a pop node
logger.info("cards_df_pop_parts['part_type_full'] = (")
cards_df_pop_parts["part_type_full"] = (
    cards_df_pop_parts["pop_type"] + "-" + cards_df_pop_parts["part_type"]
)
# -

# %% Checkings

# + deletable=false editable=false run_control={"frozen": true}
# (cards_df_pop_parts==cards_df_pop_parts2).all().all()

# + deletable=false editable=false run_control={"frozen": true}
# cards_df_pop_parts[cards_df_pop_parts['part_type']=='step_condition']['part'].unique()
# -

# %% Export
logger.info("cards_df_pop_parts.set_index(")
cards_df_pop_parts.set_index(
    ["card_id", "paragraph_order", "pop_order", "part_order"]
).to_sql(export_table_name, engine, if_exists="replace")

# %% Create metrics for pops and parts

logger.info("cards_df_pop_parts = pd.read_sql_table(export_table_name, engine)")
cards_df_pop_parts = pd.read_sql_table(export_table_name, engine)

cards_df_pop_parts["part"] = cards_df_pop_parts["part"].replace("", numpy.nan).dropna()
cards_df_pop_parts = cards_df_pop_parts.dropna(subset=["part"])

# +
# cards_df_pop_parts[cards_df_pop_parts['card_id']=='ebf1a7f3-7621-5fe7-826f-296d088df97c']
# -
logger.info(
    """cards_df_pop_parts['paragraph_pk'] = cards_df_pop_parts.progress_apply"""
)
cards_df_pop_parts["paragraph_pk"] = cards_df_pop_parts.progress_apply(
    lambda x: "-".join([x["card_id"], str(int(x["paragraph_order"]))]), axis=1
)

logger.info("""cards_df_pop_parts['pop_pk'] = cards_df_pop_parts.progress_apply""")
cards_df_pop_parts["pop_pk"] = cards_df_pop_parts.progress_apply(
    lambda x: "-".join(
        [x["card_id"], str(int(x["paragraph_order"])), str(int(x["pop_order"]))]
    ),
    axis=1,
)

cards_df_pop_parts.set_index(
    ["card_id", "paragraph_order", "pop_order", "part_order", "paragraph_pk", "pop_pk"]
).to_sql(export_table_name, engine, if_exists="replace")

# +
# cards_df_pop_parts[cards_df_pop_parts['card_id']=='ebf1a7f3-7621-5fe7-826f-296d088df97c']
# -

# %% Create pop metrics

# + hideCode=false
metrics_pop = cards_df_pop_parts.pivot_table(
    index=["card_id", "paragraph_pk", "pop_type"],
    values=["pop"],
    aggfunc=lambda x: len(x),
)
metrics_pop.columns = ["pop_count"]
metrics_pop[metrics_pop["pop_count"] > 5]
# -

metrics_pop.to_sql(export_table_name + "_metrics_pop", engine, if_exists="replace")

# %% Create part metrics

# + hideCode=false
metrics_part = cards_df_pop_parts.pivot_table(
    index=["card_id", "paragraph_pk", "pop_pk", "part_type"],
    values=["part"],
    aggfunc=lambda x: len(x),
)
metrics_part.columns = ["part_count"]
metrics_part[metrics_part["part_count"] > 3]
# -

metrics_part.to_sql(export_table_name + "_metrics_part", engine, if_exists="replace")

logger.info(f"FINISHED: {__file__}")

# %%
