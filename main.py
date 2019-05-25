from MAB_agent import Agent
from graph_utils import generate_random_graph
from settings import H, SAMPLE_SIZE, BATCH_SIZE, ALPHA, MAX_EPISODES, gamma


def run():
    graph, x_original = generate_random_graph()
    agent = Agent(H, BATCH_SIZE, ALPHA, MAX_EPISODES, gamma)
    agent.learn(graph, x_original, SAMPLE_SIZE)
    print(agent.policy)
    print (agent.rewards)


run()
