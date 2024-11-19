import subprocess
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.SimpleAgents.HybridBlueAgent import HybridBlueAgent

MAX_EPS = 100
agent_name = 'Blue'

def wrap(env):
    return ChallengeWrapper(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    # Define action space first
    action_space = [
        1,                    # monitor
        133, 134, 135, 139,  # restore actions for enterprise and opserver
        3, 4, 5, 9,          # analyse actions for enterprise and opserver
        16, 17, 18, 22,      # remove malware from enterprise and opserver
        11, 12, 13, 14,      # analyse actions for user hosts
        141, 142, 143, 144,  # restore actions for user hosts
        132,                 # restore defender system
        2,                   # analyse defender system
        15, 24, 25, 26, 27,  # remove malware from defender and user hosts
        # Add decoy actions
        69, 70, 71, 72,      # Deploy decoy on enterprise0-2 and opserver0
        73, 74, 75, 76,      # Deploy decoy on user hosts
        77                   # Deploy decoy on defender
    ]

    # Initialize agent with proper dimensions
    agent = HybridBlueAgent(
        input_dim=52,
        hidden_dim=256,
        output_dim=len(action_space)
    )
    
    # Load the trained model
    agent.load('CybORG/Checkpoints/best_model.pt')

    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    commit_hash = get_git_revision_hash()
    name = input('Name: ')
    team = input("Team: ")
    name_of_agent = input("Name of technique: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    print(f'Using agent {agent.__class__.__name__}')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:
            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)
            
            observation = wrapped_cyborg.reset()
            total_reward = []
            actions = []
            
            for i in range(MAX_EPS):
                r = []
                a = []
                for j in range(num_steps):
                    # Get action space for current step
                    current_action_space = wrapped_cyborg.get_action_space(agent_name)
                    # Get action using our action space
                    action = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                    
                    if done:
                        break
                
                # Store episode results
                total_reward.append(sum(r))
                actions.append(a)
                
                # Reset for next episode
                observation = wrapped_cyborg.reset()
                if hasattr(agent, 'end_episode'):  # Check if method exists
                    agent.end_episode()

            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
