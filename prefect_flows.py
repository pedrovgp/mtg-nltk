from prefect import Task, Flow
import logging
import os

from flows.mtgmetaio import flow_scrapy_crawl, get_flow_landing_to_decks

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

# %% FLOW FULL PIPELINE DEFINITION


class CreateCardsDatabase(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("a_create_cards_database.py")


class LoadDecksIntoDatabase(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("load_decks_into_database.py")


class EnhanceCardsDataWithoutNLP(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("b_enhance_cards_no_nlp.py")


class EnhanceCardsDataWithNLP(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("c_enhance_cards_with_nlp.py")


class BuildIndividualCardsInOutGraph(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("d_build_individual_cards_graph.py")


class BuildTextToEntityGraphs(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("e_build_text_to_entity_graphs.py")


class BuildGraphForAFewCardsAndSaveInPics(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        execfile("build_graph_of_selected_cards.py")


flow_full_data_pipeline = Flow("Imperative-MTG-NLP-full-flow")

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
flow_full_data_pipeline.set_dependencies(
    task=build_graph_for_a_few_cards_and_save_pics,
    upstream_tasks=[build_text_to_entity_graphs],
)
flow_full_data_pipeline.set_dependencies(
    task=build_text_to_entity_graphs,
    upstream_tasks=[build_individual_cards_graph],
)
flow_full_data_pipeline.set_dependencies(
    task=build_individual_cards_graph,
    upstream_tasks=[enhance_cards_with_nlp],
)

flow_full_data_pipeline.set_dependencies(
    task=enhance_cards_with_nlp,
    upstream_tasks=[enhance_cards_without_nlp],
)

flow_full_data_pipeline.set_dependencies(
    task=enhance_cards_without_nlp,
    upstream_tasks=[create_cards_database],
)

flow_full_data_pipeline.set_dependencies(
    task=load_decks_into_database,
    upstream_tasks=[create_cards_database],
)


# %% FLOW flow_deck_graph

flow_deck_graph = Flow("Imperative-build-deck-graph")

# %% INSTANTIATE TASKS IN flow_deck_graph


class TBD:
    pass


# from deck name input, returns dataframe
loadCardsAsDataframe = TBD()
# from Dataframe input, daframe with Incoming and Outgoing cards Graphs
createIndividualCardsGraph = TBD()
# from df of graphs, returns composed Graph, with nodes collpased to only cards
composeAllGraphsCollapsed = TBD()
# writes graph to pics
drawGraph = TBD()

# %% SET DEPENDENCIES in flow_deck_graph
# flow_deck_graph.set_dependencies(
# )

deck_slug = None
run_full_flow = True
run_scrapy_crawl = False
run_mtgmetaio_landind_to_deks = True
if __name__ == "__main__":

    if deck_slug:
        flow_deck_graph.visualize()
        flow_state = flow_deck_graph.run()
        flow_full_data_pipeline.visualize(flow_state=flow_state)

    if run_full_flow:
        flow_full_data_pipeline.visualize()
        flow_state = flow_full_data_pipeline.run()
        flow_full_data_pipeline.visualize(flow_state=flow_state)

    if run_scrapy_crawl:
        flow_scrapy_crawl.visualize()
        flow_state = flow_scrapy_crawl.run()
        flow_scrapy_crawl.visualize(flow_state=flow_state)

    if run_mtgmetaio_landind_to_deks:
        flow_landing_to_decks = get_flow_landing_to_decks(for_urls=[])
        flow_landing_to_decks.visualize()
        flow_state = flow_landing_to_decks.run()
        flow_landing_to_decks.visualize(flow_state=flow_state)

    pass
