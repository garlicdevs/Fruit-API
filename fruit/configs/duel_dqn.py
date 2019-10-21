import tensorflow as tf
import numpy as np
from fruit.configs.dqn import AtariDQNConfig


# Configuration for Atari games using Duel network
class AtariDuelDQNConfig(AtariDQNConfig):
    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 32, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 64, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        layer_3 = self.layer_manager.create_conv_layer(layer_2, 64, 3, strides=1, activation_fn='relu', padding='valid', scope='tf_layer_3')

        layer_3_r = tf.multiply(layer_3, np.sqrt(2))

        # State stream
        layer_4 = self.layer_manager.create_fully_connected_layer(layer_3_r , 512, activation_fn='relu', scope='tf_layer_4')
        output_1 = self.layer_manager.create_output(layer_4, 1, activation_fn='relu', scope='tf_layer_4_1')

        # Action advantage stream
        layer_5 = self.layer_manager.create_fully_connected_layer(layer_3_r, 512, activation_fn='relu', scope='tf_layer_5')
        output_2 = self.layer_manager.create_output(layer_5, self.output_size, scope='tf_layer_5_1')

        # Q values
        output = tf.add(output_1, tf.subtract(output_2, tf.expand_dims(tf.reduce_mean(output_2, axis=1), axis=1)))

        return output

    def create_train_step(self):
        with tf.name_scope('compute-clip-grads'):
            gradients = self.optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, 10)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = self.optimizer.apply_gradients(clipped_grads_tensors)
        summaries = tf.summary.merge_all()
        return summaries, train_step

    def create_loss(self):
        with tf.name_scope('estimate_q_values'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            estimate_q_values = tf.reduce_sum(tf.multiply(self.tf_output, actions_one_hot), axis=1)

        with tf.name_scope('loss'):
            loss = tf.squared_difference(estimate_q_values, self.tf_targets)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_mean(loss)
            tf.summary.scalar('total-loss', total_loss)

        return estimate_q_values, loss, total_loss

    def train(self, session, data_dict):
        states, actions, rewards, next_states, terminals, learning_rate, logging, global_step = \
            self.get_params(data_dict)

        if global_step % self.target_update == 0:
            self.update_target_network(session)

        feed_dict = {self.tf_inputs: next_states}
        q_values_next = session.run(self.tf_target_output, feed_dict=feed_dict)

        max_q_values_next = np.max(q_values_next, axis=1)
        targets = rewards + np.subtract(1., terminals) * self.gamma * max_q_values_next

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