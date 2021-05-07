# MTG Deck Builder

## Quick INSTALL AND RUN guide:
>pip install poetry  
>poetry install  
>python -m spacy download en_core_web_sm

Then, please install postgres locally and create a database 'mtg' owned by an user 'mtg' with password 'mtg'.
This is to run parts of the code more efficiently.

Now, to actually run the data processing flows:

> TODO prefect here

And, if you want to see a simplfied MVP of what a final product would look like, run
> streamlit run 04c_ga_for_deck_building.py 

and open your browser on http://localhost:8501/.


## Ideal
The ideal of this project is to, given two sets of cards and a format, return you the best deck you could build for that format with the first set of cards, to play against a deck built from the second set fo cards. Remember: ideal is not a goal (and best deck would be a very disputable term for MTG anyway). As Eduardo Galeano said, ideal is like the horizon, it is a direction for us to walk towards, but never to reach.

## Method

We could get info from the meta game and build decks from there. Or we could (somehow) get decks victory stats and learn from that.
Both approaches are valid and would profit from all the human learning in this arena.

But the approach proposed here is slightly different, the idea is to make the engine to reason more from cards relations than from human usage.
So, it tries to build features from cards relations, inferred by their text, properties and games rules; and learn some mapping from these features to a ranking of goodness of a deck. That may be learned from meta game of decks stats and history, but than the knowledge acquired by the model would be able to generalize to new cards and sets.

### Building blocks

Establishing relationships between cards starts by applying NLP to individual cards and identifying mentioned entites:

!['Terror' card out graph](pics/03a-card1out.png "'Terror' card out graph")

And on the other side, detecting entites partaining to cards:

!['Soltari Visionary' card in graph](pics/03a-card2in.png "'Soltari Visionary' card in graph")

Joining the graphs would result in a path (a established relation) between the two cards:

!['Terror' -> 'Soltari Visionary' relation](pics/03a-g1out-g2in.png "'Terror' -> 'Soltari Visionary' relation")

Actually, it describes the relation 'Terror' -> 'Soltari Visionary'. The inverse relation would look different.

Joining on the in and out graphs of all cards in a deck would result in the decks relationship graph. That's where we want to extract our features from.
This is still ongoing work (help is welcome and appreciated, I would love to discuss ideias).

Files 04a and 04b outline a few other ideas (not yet implemented).

