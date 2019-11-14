from fruit.agents.factory import AgentFactory
from fruit.envs.gym import GymEnvironment
from fruit.plugins.tf_plugin import TensorForcePlugin


def main():
    # Create a Gym environment in FruitAPI
    env = GymEnvironment(env_name='CartPole-v1')

    # Create a PPO learner
    agent = AgentFactory.create(TensorForcePlugin().get_learner(),
                                None, env, num_of_epochs=1, steps_per_epoch=5e4, log_dir='train/ppo_checkpoints',
                                checkpoint_frequency=5e4,
                                # TensorForce parameters
                                algorithm='ppo', network='auto',
                                # Optimization
                                batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
                                optimization_steps=5,
                                # Reward estimation
                                likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
                                # Critic
                                critic_network='auto',
                                critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
                                preprocessing=None,
                                # Exploration
                                exploration=0.0, variable_noise=0.0,
                                # Regularization
                                l2_regularization=0.0, entropy_regularization=0.0,
                                # TensorFlow etc
                                name='agent', device=None, parallel_interactions=1, seed=None, execution=None,
                                saver=None, summarizer=None, recorder=None
                                )

    # Train it
    agent.train()


if __name__ == '__main__':
    main()
