# Wii Sports Archery Recreation

This project aims to recreate the Wii Sports Archery game environment and to train an AI to play it. 


![What it looks like](assets/demo.mov)
# Overview 
I recreated the game in Unity in order to make an AI Agent control this Mii, with the help of the MLAgents package.
By cloning this repo, you can reuse my training environment like a Gym Environment thanks to the unity executable, as long as the `WiiSports-Archery-v1` is in the working directory

```
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

env_args = #specify the rules of the games
unity_env = UnityEnvironment("WiiSports-Archery-v1",additional_args=env_args)
env = UnityToGymWrapper(unity_env)
```

# Detailed Rules

The environment has multiple features and is customizable (2 levels of diifculty, toggle wind, ...)
Its rules are very similar to the original game from Nintendo.
    - The agent gains points depending on where his arrows lands:
        if he misses the targets, 0 points 
        if he hits  the targets from 1 point (inner ring) to 10 points (outer ring) depending on his precision
    - The agent is always at the same spot.
    - The agent has a certain number of lives (say 3 lives) before the traget changes of position.
    - In easy mode, there are 9 possible spots for the target.
    - In hard mode, the target can be anywhere on the brick wall.
    - If toggled, the wind is changing whenever the target changes. It has constant strength but random direction, and is along (XY)
    - If heuristics is toggled, the agent has an additionnal penalty for being far away from the target
The Agent choose the vector (XYZ) that represents its shooting direction;
The agent observes the position of the target and the direction of the wind.
The action space is ``` Box(-1,1,shape = (3,))``` and the observation Space is `Box(-inf,inf,shape = (6,)) ` 

Use ```demo --help``` to learn about how to change the environment parameters and see how they affect the game rendering.

# Experiments

I implemented a training script using PPO implementation from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)