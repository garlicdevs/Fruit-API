class WorldRewards(object):
    """
    WorldRewards

    Methods inherited by WorldLayer
    Has context for game settings, map state and player state

    Responsabilities:
        Add rewards to player in response to game events
    """

    def __init__(self):
        super(WorldRewards, self).__init__()

    # Reward functions implementing game modes
    def reward_battery(self):
        """
        Add a battery level reward
        """
        if not 'battery' in self.mode:
            return
        mode = self.mode['battery']
        if mode and mode and self.__test_cond(mode):
            self.logger.debug('Battery out')
            self.player.stats['reward'] += mode['reward']

            self.player.game_over = self.player.game_over or mode['terminal']

    def reward_item(self, item_type):
        """
        Add a food collision reward
        """
        assert isinstance(item_type, str)

        if not 'items' in self.mode:
            return
        mode = self.mode['items']
        if mode and mode[item_type] and self.__test_cond(mode[item_type]):
            self.logger.debug("{item_type} consumed".format(item_type=item_type))
            self.player.stats['reward'] += mode[item_type]['reward']
            self.player.stats['score'] += mode[item_type]['reward']

            self.player.game_over = self.player.game_over or mode[item_type]['terminal']

    def reward_wall(self):
        """
        Add a wall collision reward
        """
        if not 'wall' in self.mode:
            return
        mode = self.mode['wall']
        if mode and mode and self.__test_cond(mode):
            self.logger.debug("Wall {x}/{y}'".format(x=self.bumped_x, y=self.bumped_y))
            self.player.stats['reward'] += mode['reward']

            self.player.game_over = self.player.game_over or mode['terminal']

    def reward_explore(self):
        """
        Add an exploration reward
        """
        if not 'explore' in self.mode:
            return
        mode = self.mode['explore']
        if mode and mode['reward'] and self.__test_cond(mode):
            self.player.stats['reward'] += mode['reward']
            self.player.stats['score'] += mode['reward']

            self.player.game_over = self.player.game_over or mode['terminal']

    def reward_goal(self):
        """
        Add an end goal reward
        """
        if not 'goal' in self.mode:
            return
        mode = self.mode['goal']
        if mode and mode['reward'] and self.__test_cond(mode):
            if mode['reward'] > 0:
                self.logger.info("Escaped!!")
            self.player.stats['reward'] += mode['reward']
            self.player.stats['score'] += mode['reward']

            self.player.game_over = self.player.game_over or mode['terminal']

    def reward_proximity(self):
        """
        Add a wall proximity reward
        """
        if not 'proximity' in self.mode:
            return
        mode = self.mode['proximity']

        # Calculate proximity reward
        reward = 0
        for sensor in self.player.sensors:
            if sensor.sensed_type == 'wall':
                reward += sensor.proximity_norm()
            else:
                reward += 1

        reward /= len(self.player.sensors)
        #reward = min(1.0, reward * 2)
        reward = min(1.0, reward * reward)
        # TODO: Configurable bonus reward threshold. Pass extra args to `__test_cond`?
        #if mode and mode and reward > 0.75 and self.__test_cond(mode):
        if mode and mode and self.__test_cond(mode):
            # Apply bonus
            reward *= mode['reward']

        self.player.stats['reward'] += reward

    def __test_cond(self, mode):
        try:
            return mode['cond'](self)
        except KeyError:
            return True
