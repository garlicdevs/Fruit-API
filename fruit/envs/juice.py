from fruit.envs.base import BaseEnvironment
import numpy as np

########################################################
# Date: 18-Feb-2018
# Author: Duy Nguyen
# Email: garlicdevs@gmail.com
########################################################


class RewardProcessor:
    def get_reward(self, rewards):
        pass

    def clone(self):
        pass

    def get_number_of_objectives(self):
        pass


class FruitEnvironment(BaseEnvironment):
    def __init__(self, game_engine, max_episode_steps=10000, state_processor=None, reward_processor=None):
        self.game = game_engine
        self.max_episode_steps = max_episode_steps
        self.num_of_steps = 0
        self.processor = state_processor
        self.is_render = self.game.is_render()
        self.num_of_objs = game_engine.get_num_of_objectives()
        self.multi_objs = True if self.num_of_objs > 1 else False
        self.r_processor = reward_processor

        # Print info
        print("#################################################")
        print("Game environment " + self.game.get_game_name() + " is created !!!")
        print("Seed: " + str(self.game.get_seed()))
        print("#################################################")

    def clone(self):
        processor = None
        r_processor = None
        if self.processor is not None:
            processor = self.processor.clone()
        if self.r_processor is not None:
            r_processor = self.r_processor.clone()
        return FruitEnvironment(self.game.clone(), max_episode_steps=self.max_episode_steps,
                                state_processor=processor, reward_processor=r_processor)

    def get_state(self):
        if self.processor is not None:
            return self.processor.process(self.game.get_state())
        else:
            return self.game.get_state()

    def get_map(self, type=-1):
        if self.processor is not None:
            return self.processor.process(self.game.get_map(type))
        else:
            return self.game.get_map(type)

    def reset(self):
        self.num_of_steps = 0
        self.game.reset()

    def is_terminal(self):
        if self.num_of_steps > self.max_episode_steps:
            return True
        return self.game.is_terminal()

    def step(self, action):
        self.num_of_steps = self.num_of_steps + 1
        if self.r_processor is None:
            if self.multi_objs:
                return self.game.step(action)
            else:
                rewards = self.game.step(action)
                total_reward = 0
                if isinstance(rewards, (list, tuple, np.ndarray)):
                    for r in rewards:
                        total_reward = total_reward + r
                else:
                    total_reward = rewards
                return total_reward
        else:
            rewards = self.game.step(action)
            return self.r_processor.get_reward(rewards)

    def get_state_space(self):
        from fruit.types.priv import Space
        if self.processor is None:
            return self.game.get_state_space()
        else:
            min = np.zeros([84, 84], dtype=np.uint8)
            max = np.full([84, 84], 255)
            return Space(min, max, True)

    def get_action_space(self):
        from fruit.types.priv import Space
        return Space(0, len(self.game.get_action_space()) - 1, True)

    def debug(self):
        self.game.debug()

    def render(self):
        return self.game.render()

    def set_seed(self, seed):
        self.game.set_seed(seed)

    def get_current_steps(self):
        return self.num_of_steps

    def get_number_of_objectives(self):
        if self.r_processor is not None:
            return self.r_processor.get_number_of_objectives()
        else:
            return self.game.get_num_of_objectives()

    def get_number_of_agents(self):
        return self.game.get_num_of_agents()

    def get_pareto_solutions(self):
        return self.game.get_pareto_solutions()

    def get_key_pressed(self):
        return self.game.get_key_pressed()


class MAFruitEnvironment(FruitEnvironment):

    def clone(self):
        return MAFruitEnvironment(self.game.clone(), max_episode_steps=self.max_episode_steps,
                                  state_processor=self.processor.clone(), multi_objective=self.multi_objs)

    def step(self, action):
        self.num_of_steps = self.num_of_steps + 1
        if self.multi_objs:
            return self.game.step(action)
        else:
            rewards = self.game.step(action)
            total_reward = 0
            for r in rewards:
                total_reward = total_reward + r
            return total_reward

    def get_num_of_agents(self):
        return self.game.get_num_of_agents()
