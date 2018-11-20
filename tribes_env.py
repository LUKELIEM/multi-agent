import collections
import itertools
import os.path
import tkinter as tk

import gym
import gym.envs.registration
import gym.spaces

import numpy as np

# The 8 actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ROTATE_RIGHT = 4
ROTATE_LEFT = 5
LASER = 6
NOOP = 7

"""
In this file, we will implement a Gathering environment where the agents are organized by tribes with
a "Us" versus "Them" mentality.
"""
class GatheringEnv(gym.Env):

    # Some basic parameters for the Gathering Game
    metadata = {'render.modes': ['human']}
    scale = 20           # Used to scale to display during rendering

    # Viewbox is to implement partial observable Markov game
    viewbox_width = 10
    viewbox_depth = 20
    padding = max(viewbox_width // 2, viewbox_depth - 1)  # essentially 20-1=19

    # To help agents distinquish between themselves, the other agents and the apple
    agent_colors = ['red', 'red', 'yellow', 'blue', \
        'yellow', 'yellow', 'yellow', 'yellow']    # Can support up to 8 agents

    # A function to build the game space from a text file
    def _text_to_map(self, text):

        m = [list(row) for row in text.splitlines()]  # regard "\r", "\n", and "\r\n" as line boundaries 
        l = len(m[0])
        for row in m:   # Check for errors in text file
            if len(row) != l:
                raise ValueError('the rows in the map are not all the same length')

        # This essentially add a padding of 20 cells around the region enclosed by the walls or by the
        # food (if there is no wall). During rendering you will observe this padded region when a laser
        # is fired     
        def pad(a):
            return np.pad(a, self.padding + 1, 'constant')

        a = np.array(m).T
        self.initial_food = pad(a == 'O').astype(np.int)    # Pad 20 around food
        self.walls = pad(a == '#').astype(np.int)           # Pad 20 around the walls


    # This is run when the environment is created
    def __init__(self, n_agents=1, map_name='default'):

        self.n_agents = n_agents    # Set number of agents
        self.root = None            # For rendering

        # Create game space from text file
        if not os.path.exists(map_name):
            expanded = os.path.join('maps', map_name + '.txt')
            if not os.path.exists(expanded):
                raise ValueError('map not found: ' + map_name)
            map_name = expanded
        with open(map_name) as f:
            self._text_to_map(f.read().strip())    # This sets up self.initial_food and self.walls

        # Populate the rest of environment parameters
        self.width = self.initial_food.shape[0]
        self.height = self.initial_food.shape[1]

        # This is a partial observable Markov game. The observation for each agent is a stack of 4 frames of
        # 10x20 pixels (view box). These 4 frames identifies:
        #    1. Location of Food
        #    2. ???
        #    3. Location of agents
        #    4. Location of the walls
        # if and only if they are in the viewbox of the agent.
        self.state_size = self.viewbox_width * self.viewbox_depth * 4
        self.observation_space = gym.spaces.MultiDiscrete([[[0, 1]] * self.state_size] * n_agents)
        self.action_space = gym.spaces.MultiDiscrete([[0, 7]] * n_agents)   # Action space for n agents

        self._spec = gym.envs.registration.EnvSpec(**_spec)
        self.reset()    # Reset environment
        self.done = False


    # A function to check if the location the agent intends to move into will result in a collision with
    # another agent
    def _collide(self, agent_index, next_location, current_locations):

        for j, current in enumerate(current_locations):
            if j is agent_index:      # Skip its own current location
                continue
            if next_location == current:   # If the location is occupied
                # print("Collide!")
                return True

        return False



    # A function to take the game one step forward
    # Inputs: a list of actions indexed by agent
    def _step(self, action_n):

        assert len(action_n) == self.n_agents  # Error check for action list
        # Set action of tagged agents to NOOP
        action_n = [NOOP if self.tagged[i] else a for i, a in enumerate(action_n)]

        # Initialize variables for movement and for beam
        self.beams[:] = 0

        movement_n = [(0, 0) for a in action_n]

        # Update movement if action is UP, DOWN, RIGHT or LEFT
        for i, (a, orientation) in enumerate(zip(action_n, self.orientations)):
            if a not in [UP, DOWN, LEFT, RIGHT]:
                continue
            # a is relative to the agent's orientation, so add the orientation
            # before interpreting in the global coordinate system.
            #
            # This line is really not obvious to read. Replace it with something
            # clearer if you have a better idea.
            a = (a + orientation) % 4
            movement_n[i] = [
                (0, -1),  # up/forward
                (1, 0),   # right
                (0, 1),   # down/backward
                (-1, 0),  # left
            ][a]

        # The code below updates agent location based on proposed movements 
        current_locations = [a for a in self.agents] 
        for i, ((dx, dy), (x, y)) in enumerate(zip(movement_n, self.agents)):  # For each agent

            if self.tagged[i]:   # skip agents that are tagged
                continue 
            next_ = ((x + dx), (y + dy))   # Calculate next location

            if self.walls[next_]:
                next_ = (x, y)              # Do not move into walls

            # Do not move into the current location of another agent

            if self._collide(i, next_, current_locations):
                # find the first possible move that does not result in collision
                """
                for move in movement_n:
                    dx, dy = move
                    next_ = ((x + dx), (y + dy))   # Calculate possible next location
                    if not self._collide(i, next_, current_locations):
                        break
                """

                next_ = (x, y)   # If all possible moves result in collision, stay in original spot

            self.agents[i] = next_

        """
        # The code section below updates agent location based on actions that are movements    
        next_locations = [a for a in self.agents]  # Initialize next_locations
        # If a key is not found in the dictionary, then instead of a KeyError being thrown, a new entry 
        # is created.
        next_locations_map = collections.defaultdict(list)

        for i, ((dx, dy), (x, y)) in enumerate(zip(movement_n, self.agents)):  # For each agent

            if self.tagged[i]:   # skip agents that are tagged
                continue        

            next_ = ((x + dx), (y + dy))   # Calculate next location

            if self.walls[next_]:
                next_ = (x, y)              # Do not move into walls

            next_locations[i] = next_
            next_locations_map[next_].append(i)  # append agent to next_location_map

        # If there are more than 1 agent in the same location
        for overlappers in next_locations_map.values():
            if len(overlappers) > 1:
                for i in overlappers:
                    next_locations[i] = self.agents[i]  # return agent to their previous location
        self.agents = next_locations    # Update agent locations
        """

        for i, act in enumerate(action_n):
            if act == ROTATE_RIGHT:
                self.orientations[i] = (self.orientations[i] + 1) % 4
            elif act == ROTATE_LEFT:
                self.orientations[i] = (self.orientations[i] - 1) % 4
            elif act == LASER:
                self.beams[self._viewbox_slice(i, 5, 20, offset=1)] = 1


        # Prepare obs_n, reward_n, done_n and info_n to be returned        
        obs_n = self.state_n    # obs_n is self.state_n
        reward_n = [0 for _ in range(self.n_agents)]
        done_n = [self.done] * self.n_agents
        info_n = [{}] * self.n_agents


        # This is really shitty code writing. If agent lands on a food cell, that cell is set to -15.
        # Then for each subsequent step, it is incremented by 1 until it reaches 1 again.
        # self.initial_food is the game space created from the text file whereby the cell with food 
        # is given the value of 1, every other cell has the value of 0.
        self.food = (self.food + self.initial_food).clip(max=1)

        # In case the agent lands on a cell with food, or is tagged
        for i, a in enumerate(self.agents):
            if self.tagged[i]:
                continue
            if self.food[a] == 1:
                self.food[a] = -15    # Food is respawned every 15 steps once it has been consumed
                reward_n[i] = 1       # Agent is given reward of 1
            if self.beams[a]:
                self.tagged[i] = 25   # If agent is tagged, it is removed from the game for 25 steps

        # Respawn agent after 25 steps
        for i, tag in enumerate(self.tagged):
            if tag == 1:     # It is time to respawn agent i
                self.agents[i] = self.spawn_points[i]
                self.orientations[i] = UP
                self.tagged[i] = 0


        # This is where the tagged counter (25 steps) is updated
        self.tagged = [max(i - 1, 0) for i in self.tagged]  


        return obs_n, reward_n, done_n, info_n

    # Generate slice(tuple) to slice out observation space for agents
    def _viewbox_slice(self, agent_index, width, depth, offset=0):
        
        # These are inputs for generating an observation space for the agent
        # Note that if width is 10, the agent can perceive 5 pixels to the left, 
        # 1 pixel directly in front of itself, and 4 pixels to its right.
        left = width // 2
        right = left if width % 2 == 0 else left + 1
        x, y = self.agents[agent_index]

        # This is really hard-to-read code. Essentially, it generates the observation
        # spaces for an agent in all 4 orientations, then only return the one indexed
        # by its current orientation.
        # Note: itertools.starmap maps the orientation-indexed tuple to slice()
        return tuple(itertools.starmap(slice, (
            ((x - left, x + right), (y - offset, y - offset - depth, -1)),      # up
            ((x + offset, x + offset + depth), (y - left, y + right)),          # right
            ((x + left, x - right, -1), (y + offset, y + offset + depth)),      # down
            ((x - offset, x - offset - depth, -1), (y + left, y - right, -1)),  # left
        )[self.orientations[agent_index]]))


    # state_n (next state) is a property object. So this function is run everytime state_n is
    # called as a varaiable.
    @property
    def state_n(self):

        # Create a game space for agent locating
        agents = np.zeros_like(self.food)  

        # self.agent is a list of agent locations indexed by agent index
        for i, loc in enumerate(self.agents):    
            if not self.tagged[i]:
                agents[loc] = 1     # Mark the agent's location

        food = self.food.clip(min=0)   # Mark the food's location

        # Zero out next states for the agents
        s = np.zeros((self.n_agents, self.viewbox_width, self.viewbox_depth, 4))

        # Enumerate index, (agent orientation, agent location) by agent index
        for i, (orientation, (x, y)) in enumerate(zip(self.orientations, self.agents)):

            if self.tagged[i]:
                continue     # Skip if agent has been tagged out of the game

            # If agent is not tagged, ....

            # Construct the full state for the game, which consists of:
            # 1. Location of Food
            # 2. ???
            # 3. Location of agents
            # 4. Location of the walls
            full_state = np.stack([food, np.zeros_like(food), agents, self.walls], axis=-1)
            full_state[x, y, 2] = 0   # Zero out the agent's location ???

            # Create observation space for learning agent using _viewbox_slice()
            xs, ys = self._viewbox_slice(i, self.viewbox_width, self.viewbox_depth)
            observation = full_state[xs, ys, :]

            # Orient the observation space correctly
            s[i] = observation if orientation in [UP, DOWN] else observation.transpose(1, 0, 2)

        return s.reshape((self.n_agents, self.state_size))  # Return the agents' observations


    # To reset the environment
    def _reset(self):

        # Build food stash
        self.food = self.initial_food.copy()


        # Rebuild the wall (by subtracting padding from self.walls - very weird implementation!!!)
        # Essentially, think of a much larger game area (+20 cells on each side) surrounding the 
        # walled region.
        p = self.padding
        self.walls[p:-p, p] = 1
        self.walls[p:-p, -p - 1] = 1
        self.walls[p, p:-p] = 1
        self.walls[-p - 1, p:-p] = 1

        self.beams = np.zeros_like(self.food)  # self.beams region is as big as self.food (weird!)

        # Set up agent parameters
        # The agents are spawned at the right upper corner of the game area, one next to the other
        self.agents = [(i + self.padding + 1, self.padding + 1) for i in range(self.n_agents)]
        self.spawn_points = list(self.agents)
        self.orientations = [UP for _ in self.agents]   # Orientation = UP
        self.tagged = [0 for _ in self.agents]          # Tagged = False

        return self.state_n  # Since state_n is a property object, so it will call function _state_n()


    # To close the rendering window
    def _close_view(self):
        # If rendering window is active, close it
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
        self.done = True   # The episode is done
    

    # TO render the game    
    def _render(self, mode='human', close=False):
        if close:
            self._close_view()
            return

        canvas_width = self.width * self.scale
        canvas_height = self.height * self.scale

        if self.root is None:
            self.root = tk.Tk()
            self.root.title('Gathering')
            self.root.protocol('WM_DELETE_WINDOW', self._close_view)
            self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
            self.canvas.pack()

        self.canvas.delete(tk.ALL)
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill='black')

        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * self.scale,
                y * self.scale,
                (x + 1) * self.scale,
                (y + 1) * self.scale,
                fill=color,
            )

        for x in range(self.width):
            for y in range(self.height):
                if self.beams[x, y] == 1:
                    fill_cell(x, y, 'yellow')
                if self.food[x, y] == 1:
                    fill_cell(x, y, 'green')
                if self.walls[x, y] == 1:
                    fill_cell(x, y, 'grey')

        for i, (x, y) in enumerate(self.agents):
            if not self.tagged[i]:
                fill_cell(x, y, self.agent_colors[i])

        if True:
            # Debug view: see the first player's viewbox perspective.
            p1_state = self.state_n[0].reshape(self.viewbox_width, self.viewbox_depth, 4)
            for x in range(self.viewbox_width):
                for y in range(self.viewbox_depth):
                    food, me, other, wall = p1_state[x, y]
                    assert sum((food, me, other, wall)) <= 1
                    y_ = self.viewbox_depth - y - 1
                    if food:
                        fill_cell(x, y_, 'green')
                    elif me:
                        fill_cell(x, y_, 'cyan')
                    elif other:
                        fill_cell(x, y_, 'red')
                    elif wall:
                        fill_cell(x, y_, 'gray')
            self.canvas.create_rectangle(
                0,
                0,
                self.viewbox_width * self.scale,
                self.viewbox_depth * self.scale,
                outline='blue',
            )

        self.root.update()


    # To close the environment
    def _close(self):
        self._close_view()

    # To delete the environment
    def __del__(self):
        self.close()


_spec = {
    'id': 'Gathering-Luke-v031',
    'entry_point': GatheringEnv,
    'reward_threshold': 500,   # The environment threshold at 100 appears to be too low
}


gym.envs.registration.register(**_spec)
