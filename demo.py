from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from argparser import parse_env_args

def run_random_rollout(env,num_episodes = 20):
    obs = env.reset()
    for _ in range(num_episodes):
        done = False
        while not done:
            action = env.action_space.sample()  # Take a random action
            obs, reward, done, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
        obs = env.reset()


if __name__ == "__main__":
    env_args = parse_env_args()
    print(f"Running a random rollout of the environment with settings: {env_args}")
    
    # Initialize the Unity environment
    unity_env = UnityEnvironment("WiiSports-Archery-v1", additional_args=env_args)
    env = UnityToGymWrapper(unity_env)
    
    # Run random rollout
    run_random_rollout(env)
    
    # Close the environment
    env.close()
    print("Demo completed")

    
