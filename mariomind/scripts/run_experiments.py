from mariomind.training.experiment_runner import run_experiments
from mariomind.utils.config import TrainConfig

if __name__ == "__main__":
    grid = {
        "agent_type": ["random", "always_right", "reflex", "dqn", "ddqn"],
        "frame_stack": [1, 2, 4],
        "reward_function": ["sparse", "progress_only", "progress_death", "default"],
        "action_space": ["RIGHT_ONLY", "SIMPLE_MOVEMENT", "CUSTOM_SMALL_ACTION_SPACE"],
        "epsilon_decay_steps": [10000, 50000, 100000],
        "replay_buffer_size": [5000, 20000, 50000],
    }
    run_experiments(TrainConfig(episodes=5), grid)
