import argparse
from pprint import pprint

def parse_env_args():
    """
    Parse the arguments so that they are ready to be fed to UnityEnvironment("WiiSports-Archery-v1",additional_args=env_args)
    """

    parser = argparse.ArgumentParser(description="Run Unity environment with different parameters and perform random rollouts.")
    
    parser.add_argument("--no_graphics", type=bool, default=True, help="Disable graphics.")

    parser.add_argument("--timescale", type=str, default="1", help="Timescale for the environment (1-100).")

    parser.add_argument("--mode", type=str, default="hard", help="Mode of the game (easy/hard).")
    
    parser.add_argument("--wind", type=str, default="on", help="Wind setting (on/off).")

    parser.add_argument("--lives", type=str, default="1", help="Number of lives in the game.")

    parser.add_argument("--heuristics", type=str, default="off", help="Heuristics setting (on/off).")
    
    assert(args.wind == "on" or args.wind == "off", "Wind must be 'on' or 'off'.")
    assert(int(args.timescale) >= 1 and int(args.timescale) <= 100, "Timescale must be an int between 1 and 100.")
    assert(args.mode == "easy" or args.mode == "hard", "Mode must be 'easy' or 'hard'.")
    assert(isinstance(int(args.lives), int) , "Lives must be an int.")
    assert(args.heuristics == "on" or args.heuristics == "off", "Heuristics must be 'on' or 'off'.")
    
    
    args = parser.parse_args()

    env_args = [
        "-timescale", args.timescale, 
        "-mode", args.mode, 
        "-wind", args.wind, 
        "-lives", args.lives,
        "-heuristics", args.heuristics
    ]
    print("-"*50)
    pprint("env_args")
    pprint(env_args)
    print("-"*50)

    return(env_args)

def parse_env_args_and_training_args():
    """
    Parse the training arguments so that they are ready to be fed to PPO
    """

    parser = argparse.ArgumentParser(description=" Run PPO with different parameters and different environment rules.")
    
    parser.add_argument("--no_graphics", type=bool, default=True, help="Disable graphics.")

    parser.add_argument("--timescale", type=str, default="100", help="Timescale for the environment (1-100).")

    parser.add_argument("--mode", type=str, default="hard", help="Mode of the game (easy/hard).")

    parser.add_argument("--wind", type=str, default="on", help="Wind setting (on/off).")

    parser.add_argument("--lives", type=str, default="1", help="Number of lives in the game.")

    parser.add_argument("--heuristics", type=str, default="off", help="Heuristics setting (on/off).")

    parser.add_argument("--save_path", type=str, default="model.pkl", help="Path to save the model.")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")

    parser.add_argument("--n_steps", type=int, default=64, help="Number of steps to run for each environment per update.")

    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")

    parser.add_argument("--device", type=str, default="cuda", help="Device to run the training on (cuda/cpu).")

    parser.add_argument("--seed", type=int, default=0, help="Seed for the random number generator.")

    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total number of timesteps to train for.")

    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging.")

    parser.add_argument("--progress_bar", type=bool, default=True, help="Show a progress bar.")

    args = parser.parse_args()

    assert(args.wind == "on" or args.wind == "off", "Wind must be 'on' or 'off'.")
    assert(int(args.timescale) >= 1 and int(args.timescale) <= 100, "Timescale must be an int between 1 and 100.")
    assert(args.mode == "easy" or args.mode == "hard", "Mode must be 'easy' or 'hard'.")
    assert(isinstance(int(args.lives), int) , "Lives must be an int.")
    assert(args.heuristics == "on" or args.heuristics == "off", "Heuristics must be 'on' or 'off'.")
    
    training_args = {
        "save_path": args.save_path,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "log_interval": args.log_interval,
        "progress_bar": args.progress_bar,
        "no_graphics": args.no_graphics
    }

    env_args = [
        "-timescale", args.timescale, 
        "-mode", args.mode, 
        "-wind", args.wind, 
        "-lives", args.lives
    ]
    print("-"*50)
    pprint("training_args")
    pprint(training_args)
    print("-"*50)  
    pprint("env_args")
    pprint(env_args)
    print("-"*50)
    return env_args,training_args
