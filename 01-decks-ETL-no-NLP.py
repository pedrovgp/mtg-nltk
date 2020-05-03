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

# # ETL of all decks

# The idea here is to
#
# 1. Get the decks as txt, build a dataframe of it (every card is an index)
# 2. Upload to database
#
# **DESIRED RESULT**:
# table = idx | card_name | deck_name 

import json
import pandas as pd
import re
from collections import defaultdict

from slugify import slugify

#from tqdm import tqdm
#tqdm.pandas()
from tqdm.notebook import tqdm_notebook
# tqdm_notebook.pandas()



# # Params

from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://mtg:mtg@localhost:5432/mtg')
engine.connect()

decks_dir = './decks/'

decks_table_name = 'decks'

# + deletable=false editable=false run_control={"frozen": true}
# # Drop table to reset it
# #engine.execute('DROP TABLE {0}'.format('public.'+decks_table_name))
# -

try:
    registered_decks = pd.read_sql_query('SELECT DISTINCT deck_id from {0}'.format('public.'+decks_table_name), 
                                     engine)
    registered_decks = registered_decks['deck_id'].values
except Exception:
    registered_decks = []

# ## Helping functions

# + code_folding=[0]
# Split dataframelist
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

# # Fetching decks

import requests
from bs4 import BeautifulSoup as BS

pg=requests.get('https://magic.wizards.com/en/articles/archive/top-decks/team-trios-constructed-pro-tour-modern-and-legacy-2018-08-02')
pg

# +
# pg = type('obj', (object,), {'text' : 'propertyValue'})()
# with open('./teste.html', 'rb') as f:
#     pg.text = f.read()
# -

soup = BS(pg.text, 'html.parser')

decks = soup.find_all(class_='deck-group')
# Main deck cards are in class="sorted-by-overview-container sortedContainer"
# Side board cards in class="sorted-by-sideboard-container  clearfix element"

for deck in tqdm_notebook(decks):

    deck_id = slugify(deck.attrs['id'], separator='_')
    if deck_id in registered_decks:
        print('Already registered, skipping')
        continue
    
    deck_name = deck.find_next('h4').text
    
    # Main deck
    main_deck_list = deck.find_next(class_="sorted-by-overview-container")
    main_cards_count = [int(x.text) for x in main_deck_list.find_all(class_='card-count')]
    main_cards_name = [x.text for x in main_deck_list.find_all(class_='card-name')]

    main_deck_list = []
    for cop, name in zip(main_cards_count, main_cards_name):
        main_deck_list.extend([name for x in range(cop)])

    main_deck_df = pd.DataFrame(main_deck_list, columns=['card_name'])
    main_deck_df['in'] = 'MAIN'
    
    # Side deck
    side_deck_list = deck.find_next(class_="sorted-by-sideboard-container")
    side_cards_count = [int(x.text) for x in side_deck_list.find_all(class_='card-count')]
    side_cards_name = [x.text for x in side_deck_list.find_all(class_='card-name')]

    side_deck_list = []
    for cop, name in zip(side_cards_count, side_cards_name):
        side_deck_list.extend([name for x in range(cop)])

    side_deck_df = pd.DataFrame(side_deck_list, columns=['card_name'])
    side_deck_df['in'] = 'SIDEBOARD'
    
    deck_df = pd.concat([main_deck_df, side_deck_df], sort=False)
    
    deck_df['deck_id'], deck_df['deck_name'] = deck_id, deck_name
    deck_df.index.rename('card_id_in_deck', inplace=True)
    deck_df = deck_df.reset_index().set_index(['card_id_in_deck', 'deck_id'])
    
    deck_df.to_sql(decks_table_name, engine, if_exists='append')

# # Create dataframe of cards

# ## Create tables for deck

import os

deck_regex = r'^(?P<amount>\d+) (?P<card_name>.*?)\n'

all_cards_names_in_decks = []
for path, dir, filenames in os.walk('./decks/'):
    for i, filename in enumerate(filenames):
        
        print('{0}/{1} decks: {2}'.format(i+1, len(filenames), filename.split('.')[0]))
        deck_name, extension = filename.split('.')[0], filename.split('.')[1]
        if extension != 'txt':
            print('SORRY, I only accept decks with .txt extensions.')
            continue
            
        deck_id = slugify(deck_name, separator='_')
        if deck_id in registered_decks:
            print('Already registered, skipping')
            continue
        
        with open(os.path.join(path, filename), 'r') as f:
            txt = f.readlines()
            #print(txt)
            deck_list = []
            for x in txt:
                if x in ['', '\n', 'SB']:
                    break
                deck_list.extend(re.findall(deck_regex, x))
        #deck_list # -> [(amount, card_name), (amount, card_name), ...]
        cards_in_deck_names_list = []
        for amount, card in deck_list:
            for j in range(int(amount)):
                cards_in_deck_names_list.append(card)
                all_cards_names_in_decks+=cards_in_deck_names_list
                
        # Export
        deck_df = pd.DataFrame(cards_in_deck_names_list)
        deck_df.columns = ['card_name']
        deck_df['deck_name'] = deck_name
        deck_df['deck_id'] = deck_id
        deck_df['in'] = 'MAIN'
        
        deck_df.index.rename('card_id_in_deck', inplace=True)
        deck_df = deck_df.reset_index().set_index(['card_id_in_deck', 'deck_id'])
        
        print('Exporting')
        deck_df.to_sql(decks_table_name, engine, if_exists='append')


