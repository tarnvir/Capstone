from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

sg = EnterpriseScenarioGenerator()
cyborg = CybORG(scenario_generator=sg)
from CybORG.Agents.Wrappers import BlueFlatWrapper

env = BlueFlatWrapper(env=cyborg)
obs, _ = env.reset()

# optional pretty-printing
from rich import print

print('Space:', env.observation_space('blue_agent_0'), '\n')
print('Observation:', obs['blue_agent_0'])
