from fruit.configs.dqn import AtariDQNConfig
import numpy as np


# Configuration for Atari games using Double Q-Network
class AtariDoubleDQNConfig(AtariDQNConfig):
    def train(self, session, data_dict):
        states, actions, rewards, next_states, terminals, learning_rate, logging, global_step = \
            self.get_params(data_dict)

        if global_step % self.target_update == 0:
            self.update_target_network(session)

        batch_size = np.array(states).shape[0]

        feed_dict = {self.tf_inputs: next_states}
        q_values_next = session.run(self.tf_output, feed_dict=feed_dict)
        max_estimate_actions = np.argmax(q_values_next, axis=1)

        q_values_target_next = session.run(self.tf_target_output, feed_dict=feed_dict)

        target_q_values_next = q_values_target_next[np.arange(batch_size), max_estimate_actions]

        targets = rewards + np.subtract(1., terminals) * self.gamma * target_q_values_next

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_targets: targets}

        if logging:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                q_values, loss, total_loss = \
                    session.run([self.tf_est_q_values, self.tf_loss, self.tf_total_loss], feed_dict=feed_dict)
                print("Q VALUES:")
                print(q_values)
                print("LOSS:")
                print(loss)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)