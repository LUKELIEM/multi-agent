import pickle
import numpy as np

import torch
from torch.autograd import Variable

"""
dir_name = 'MA_models/MA_8L0R_CR=0.1/'
episodes = 800  # This is used to recall a model file trained to a # of episodes

# There will be 8 agents - 8 AI agents, 0 random agents
num_ai_agents = 8
num_rdn_agents = 0
num_agents = num_ai_agents+num_rdn_agents  # just the sum of the two

"""

dir_name = 'MA_models/MA_8L1R_CR=0.1/'
episodes = 2000  # This is used to recall a model file trained to a # of episodes

# There will be 9 agents - 2 teams of 4 AI agents each and 1 random agents
num_ai_agents = 8
num_rdn_agents = 1
num_agents = num_ai_agents+num_rdn_agents  # just the sum of the two


# Data structure for AI agents (agents will form their own Class later on)
agents = []
actions = []
tags = []
rewards = []

# Initialize environment
render = True
num_actions = 8                       # There are 8 actions defined in Gathering

# Initialize constants
num_frames = 4
max_episodes = 1
max_frames = 500

def unpack_env_obs(env_obs):
    """
    Gathering is a partially-observable Markov Game. env_obs returned by GatheringEnv is a numpy 
    array of dimension (num_agent, 800), which represents the agents' observations of the game.

    The 800 elements (view_box) encodes 4 layers of 10x20 pixels frames in the format:
    (viewbox_width, viewbox_depth, 4).
    
    This code reshapes the above into stacked frames that can be accepted by the Policy class:
    (batch_idx, in_channel, width, height)
    
    """
    
    num_agents = len(env_obs)  # environ observations is a list of agents' observations
    
    obs = []
    for i in range(num_agents):
        x = env_obs[i]   # take the indexed agent's observation
        x = torch.Tensor(x)   # Convert to tensor
        
        # Policy is a 3-layer CNN
        x = x.view(1, 10, 20, -1)  # reshape into environment defined stacked frames
        x = x.permute(0, 3, 1, 2)  # permute to Policy accepted stacked frames
        obs.append(x)
        
    return obs  # return a list of Policy accepted stacked frames (tensor)


"""
For now, we do not implement LSTM            
# LSTM Change: Need to cycle hx and cx thru function
def select_action(model, state, lstm_hc, cuda):
    hx , cx = lstm_hc 
    num_frames, height, width = state.shape
    state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

    if cuda:
        state = state.cuda()

    probs, value, (hx, cx) = model((Variable(state), (hx, cx)))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    # LSTM Change: Need to cycle hx and cx thru function
    return action.data[0], log_prob, value, (hx, cx)
"""

def select_action(model, obs, cuda):
    """
    This code expects obs to be an array of stacked frames of the following dim:
    (batch_idx, in_channel, width, height)
    
    This is inputted into model - the agent's Policy, which outputs a probability 
    distribution over available actions.
    
    Policy gradient is implemented using torch.distributions.Categorical. 
    """
    
    # Policy is a 3-layer CNN
    # _, num_frames, width, height = obs.shape
    # obs = torch.FloatTensor(obs.reshape(-1, num_frames, width, height))
    
    # Policy is a 2-layer NN for now
    # obs = obs.view(1, -1)
   
    if cuda:
        obs = obs.cuda()
      
    probs = model(obs)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)

    return action.item(), log_prob 


def load_info(agents, narrate=False):
    for i in range(num_agents):    
        agents[i].load_info(info[i])
        if narrate:
            if agents[i].tagged:
                print('frame {}, agent{} is tagged'.format(frame,i))
            if agents[i].laser_fired:
                print('frame {}, agent{} fires its laser'.format(frame,i))
                print('and hit {} US and {} THEM'.format(agents[i].US_hit, agents[i].THEM_hit))
    return

# Load models for AI agents
if episodes > 0:
    agents= [[] for i in range(num_ai_agents)]
    # If episodes is provided (not 0), load the model for each AI agent
    for i in range(num_ai_agents):
        model_file = dir_name+'MA{}_Gather_ep_{}.p'.format(i,episodes)
        try:
            with open(model_file, 'rb') as f:
                # Model File include both model and optim parameters
                saved_model = pickle.load(f)
                agents[i], _ = saved_model
                print("Load saved model for agent {}".format(i))
        except OSError:
            print('Model file not found.')
            raise
else:
    # If episodes=0, start with a freshly initialized model for each AI agent
    for i in range(num_ai_agents):
        print("Load AI agent {}".format(i))
        agents.append(Policy(num_frames, num_actions, i))

# Load random agents    
for i in range(num_ai_agents,num_agents):
    print("Load random agent {}".format(i))
    agents.append(Rdn_Policy())

# Initialize AI and random agent data
for i in range(num_agents):
    actions = [0 for i in range(num_agents)]
    tags = [0 for i in range(num_agents)]
    rewards = [0 for i in range(num_agents)]

# Establish tribal association
tribes = []
tribes.append(Tribe(name='Vikings',color='blue', agents=[agents[0], agents[1], agents[2]]))
tribes.append(Tribe(name='Saxons', color='red', agents=[agents[3], agents[4]]))
tribes.append(Tribe(name='Franks', color='purple', agents=[agents[5], agents[6], agents[7]]))
tribes.append(Tribe(name='Crazies', color='yellow', agents=[agents[8]]))   # random agents are crazy!!!

# 9 agents in 4 tribes, used map defined in default.txt
agent_colors = [agent.color for agent in agents]
agent_tribes = [agent.tribe for agent in agents]
env = GatheringEnv(n_agents=num_agents,agent_colors=agent_colors, agent_tribes=agent_tribes, map_name='default')    
    
for ep in range(max_episodes):

    env_obs = env.reset()  # Environment return observations
    """
    # For Debug only
    print (len(agents_obs))
    print (agents_obs[0].shape)
    """
    
    # Unpack observations into data structure compatible with agent Policy
    agents_obs = unpack_env_obs(env_obs)
    
    for i in range(num_ai_agents):    # Reset agent info - laser tag statistics
        agents[i].reset_info()    
    
    env.render()
    time.sleep(1/15)  # Change speed of video rendering
    
    """
    # For Debug only
    print (len(agents_obs))
    print (agents_obs[0].shape)
    """
    
    """
    For now, we do not stack observations, and we do not implement LSTM
    
    state = np.stack([state]*num_frames)

    # Reset LSTM hidden units when episode begins
    cx = Variable(torch.zeros(1, 256))
    hx = Variable(torch.zeros(1, 256))
    """

    for frame in range(max_frames):

        for i in range(num_ai_agents):    # For AI agents
            actions[i], _ = select_action(agents[i], agents_obs[i], cuda=False)
            if actions[i] is 6:  # action[i] is a tensor, .item() returns the integer
                tags[i] += 1   # record a tag for accessing aggressiveness
                
        for i in range(num_ai_agents, num_agents):   # For random agents
            actions[i] = agents[i].select_action(agents_obs[i])
            if actions[i] is 6:
                tags[i] += 1   # record a tag for accessing aggressiveness
        
        """
        For now, we do not implement LSTM
        # Select action
        action, log_prob, state_value, (hx,cx)  = select_action(model, state, (hx,cx))        
        """

        # if frame % 10 == 0:
        #     print (actions)    
            
        # Perform step        
        env_obs, reward, done, info = env.step(actions)
        
        """
        For Debug only
        print (env_obs)
        print (reward)
        print (done) 
        """
        for i in range(num_agents):
            rewards[i] += reward[i]    # Accumulate rewards for each agent

        
        # Unpack observations into data structure compatible with agent Policy
        agents_obs = unpack_env_obs(env_obs)
        load_info(agents, narrate=True)   # Load agent info for AI agents
            
        """
        For now, we do not stack observation, may come in handy later on
        
        # Evict oldest diff add new diff to state
        next_state = np.stack([next_state]*num_frames)
        next_state[1:, :, :] = state[:-1, :, :]
        state = next_state
        """
        
        env.render()
        time.sleep(1/15)  # Change speed of video rendering

        if any(done):
            print("Done after {} frames".format(frame))
            break

env.close()  # Close the rendering window

# Print out statistics of all agents
for i in range(num_agents):
    print ("Agent{} aggressiveness is {:.2f}".format(i, tags[i]/frame))
    print ("Agent{} reward is {:d}".format(i, rewards[i]))