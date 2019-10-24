from fruit.configs.a3c import AtariA3CConfig
import tensorflow as tf
from fruit.utils.image import blacken_image


class DQAtariA3CConfig(AtariA3CConfig):
    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.shape)
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_inputs_norm = self.create_input()

        with tf.name_scope('input_2'):
            self.tf_inputs_2, self.tf_actions_2, self.tf_rewards_2, self.tf_inputs_norm_2 = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.variable_scope('network_2'):
            self.tf_actor_output_2, self.tf_critic_output_2 = self.create_network()
            self.tf_network_variables_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_2')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def predict(self, session, state):
        feed_dict = {self.tf_inputs: state}
        probs_1 = session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

        for i in range(len(state[0])):
            blacken_image(state[0][i], 84, 84, 0, 42, 0, 84)

        feed_dict_2 = {self.tf_inputs: state}
        probs_2 = session.run(self.tf_actor_output_2, feed_dict=feed_dict_2)[0]

        return [probs_1, probs_2]

    def update_dq_network(self, session):
        pred_params = self.tf_network_variables
        pred_params = sorted(pred_params, key=lambda v: v.name)
        target_params = self.tf_network_variables_2
        target_params = sorted(target_params, key=lambda v: v.name)
        ops = []
        for p1, p2 in zip(pred_params, target_params):
            op = p2.assign(p1)
            ops.append(op)
        session.run(ops)