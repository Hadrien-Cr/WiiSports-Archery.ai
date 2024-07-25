
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from argparser import parse_env_args_and_training_args
save_path = "model.pkl"

def main(env_args, training_args):
    # Initialize the Unity environment
    unity_env = UnityEnvironment("WiiSports-Archery-v1",additional_args=env_args, no_graphics=training_args["no_graphics"])
    env = UnityToGymWrapper(unity_env)

    # Initialize the agent
    model = PPO("MlpPolicy", 
                env = env,                
                batch_size=training_args["batch_size"],
                n_steps=training_args["n_steps"],
                learning_rate=training_args["learning_rate"],
                device=training_args["device"],
                gamma = 1,
                normalize_advantage=True,
                policy_kwargs=dict(
                    log_std_init=-2,
                    ortho_init=False,
                    net_arch=dict(pi=[64, 64], vf=[64, 64])
                  ))
    
    # Configure the logger
    new_logger = configure('./logs', ["stdout", "csv", "tensorboard"])
    
    # Set the logger for the model
    model.set_logger(new_logger)
    
    # Train the model
    model.learn(total_timesteps=training_args["total_timesteps"], log_interval=training_args["log_interval"], progress_bar=training_args["progress_bar"])
    
    # Save the model
    model.save(save_path)
    print("Model saved as " + save_path)


if __name__ == "__main__":
    env_args,training_args = parse_env_args_and_training_args()
    main(env_args=env_args, training_args=training_args)
