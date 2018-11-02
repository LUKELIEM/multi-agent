import random
import time

from env import GatheringEnv

from model import Policy

env = GatheringEnv(n_agents=2, map_name='default')  # Change map here

# Env API is similar to that of OpenAI Gym
state_n = env.reset()
env.render()

# Load previously trained policy agent
policy = Policy(env.state_size)
policy.load_weights()

# Render for 10000 steps
for _ in range(10000):
    state_n, reward_n, done_n, info_n = env.step([
        policy.select_action(state_n[0]),  # Agent 1 is policy AI
        random.randrange(0, 8),            # Agent 2 is random
    ])
    if any(done_n):
        break
    env.render()
    time.sleep(2/30)  # Change speed of video rendering
