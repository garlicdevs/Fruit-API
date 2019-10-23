from fruit.learners.dqn import DQNLearner


class MODQNLearner(DQNLearner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 batch_size=32, warmup_steps=10000, training_frequency=1, experience_replay_size=100000,
                 epsilon_annealing_start=1, epsilon_annealing_end=0, reward_clip_thresholds=(-1, 1)
                 ):

        super().__init__(agent=agent, name=name, environment=environment, network=network, global_dict=global_dict,
                         report_frequency=report_frequency, batch_size=batch_size, warmup_steps=warmup_steps,
                         training_frequency=training_frequency, experience_replay_size=experience_replay_size,
                         epsilon_annealing_start=epsilon_annealing_start, epsilon_annealing_end=epsilon_annealing_end,
                         epsilon_annealing_steps=agent.max_training_steps-100,
                         reward_clip_thresholds=reward_clip_thresholds)
