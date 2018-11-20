Multi-agent RL environment, in the style of DeepMind's [Gathering game](https://deepmind.com/blog/understanding-agent-cooperation/).

--- 

<video src="videos/MA_CNN_8L1R_CR=0.1-2018-11-20_11.57.36.mp4" width="320" height="200" controls preload></video>

The video shows 8 independently trained AI agents playing with 1 random agent.


The map is customizable. All of the maps from DeepMind's [paper](https://arxiv.org/pdf/1707.06600.pdf) are available in the `maps` directory. To use e.g. the "Basic single-entrance region map" from the paper, pass `map_name='region_single_entrance'` to the `GatheringEnv` constructor. To specify your own map, create a text file in the same format as the existing maps, and pass `map_name='path/to/my/custom/map'` to the `GatheringEnv` constructor.

### Installation

`pip install -r requirements.txt`
