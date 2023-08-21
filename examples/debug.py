import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from pprint import pprint


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():
    env = habitat.Env(
        config=habitat.get_config("configs/tasks/pointnav.yaml")
    )
    observations = env.reset()
    
    initial_state = env._sim.get_agent_state()
    print("initial state")
    pprint(initial_state)
    action = "MOVE_FORWARD"
    observations = env.step(action)
    end_state = env._sim.get_agent_state()
    print("end state")
    pprint(end_state)
    

if __name__ == "__main__":
    example()