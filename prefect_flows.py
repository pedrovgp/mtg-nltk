from prefect import Task, Flow
import logging

logger = logging.getLogger()

# def my_state_handler(obj, old_state, new_state):
#     msg = "\nCalling my custom state handler on {0}:\n{1} to {2}\n"
#     print(msg.format(obj, old_state, new_state))
#     return new_state


# my_flow = Flow(
#     name="state-handler-demo", tasks=[Task()], state_handlers=[my_state_handler]
# )
# my_flow.run()

def execfile(fn):
    exec(open(fn).read(), globals(), globals())

#######################


class CreateCardsDatabase(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("00.01-cards-sets.py")


class LoadDecksIntoDatabase(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("01-decks-ETL-no-NLP.py")


class EnhanceCardsDataWithoutNLP(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("01.01-cards-ETL-no-NLP.py")


class EnhanceCardsDataWithNLP(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("01.02-cards-ETL-token-nodes-edges-tables.py")


class BuildIndividualCardsInOutGraph(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("02-cards-inout-graph-builder.py")


class BuildTextToEntityGraphs(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("03-text-to-entity-path-builder.py")


class BuildGraphForAFewCardsAndSaveInPics(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("03b-build_graph_of_selected_cards.py")


# %% FLOW DEFINITION
flow = Flow("Imperative-MTG-NLP-full-flow")

# %% INSTANTIATE TASKS

# Task in sequence
create_cards_database = CreateCardsDatabase()
load_decks_into_database = LoadDecksIntoDatabase()
enhance_cards_without_nlp = EnhanceCardsDataWithoutNLP()
enhance_cards_with_nlp = EnhanceCardsDataWithNLP()
build_individual_cards_graph = BuildIndividualCardsInOutGraph()
build_text_to_entity_graphs = BuildTextToEntityGraphs()
build_graph_for_a_few_cards_and_save_pics = BuildGraphForAFewCardsAndSaveInPics()

# %% SET DEPENDENCIES
flow.set_dependencies(
    task=build_graph_for_a_few_cards_and_save_pics,
    upstream_tasks=[build_text_to_entity_graphs],
)
flow.set_dependencies(
    task=build_text_to_entity_graphs,
    upstream_tasks=[build_individual_cards_graph],
)
flow.set_dependencies(
    task=build_individual_cards_graph,
    upstream_tasks=[enhance_cards_with_nlp],
)

flow.set_dependencies(
    task=enhance_cards_with_nlp,
    upstream_tasks=[enhance_cards_without_nlp],
)

flow.set_dependencies(
    task=enhance_cards_without_nlp,
    upstream_tasks=[create_cards_database],
)

flow.set_dependencies(
    task=load_decks_into_database,
    upstream_tasks=[create_cards_database],
)

# %% VISUALIZE, RUN, WHATEVER
flow.visualize()
# flow.run()

if __name__ == "__main__":
    flow_state = flow.run()
    flow.visualize(flow_state=flow_state)

    pass
