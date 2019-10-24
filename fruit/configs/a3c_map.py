from fruit.configs.a3c import A3CConfig
import tensorflow as tf


class A3CMapConfig(A3CConfig):
    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.get_shape())
        self.output_size = len(self.action_range)
        self.map_shape = [self.history_length] + list(self.state_space.get_shape())

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_inputs_norm, \
            self.tf_map_inputs, self.tf_map_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output, self.tf_combined_layer, self.layer_3, self.layer_6 = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")

        map_inputs = self.layer_manager.create_input(tf.uint8, [None] + self.map_shape, name="tf_map_inputs")
        map_inputs_norm = tf.cast(tf.transpose(map_inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        return inputs, actions, rewards, inputs_norm, map_inputs, map_inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        # layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3')

        layer_4 = self.layer_manager.create_conv_layer(self.tf_map_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_4')
        layer_5 = self.layer_manager.create_conv_layer(layer_4, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_5')
        # layer_6 = self.layer_manager.create_fully_connected_layer(layer_5, 256, activation_fn='relu', scope='tf_layer_6')

        layer_7 = tf.concat([layer_2, layer_5], axis=3)
        layer_8 = self.layer_manager.create_fully_connected_layer(layer_7, 256, activation_fn='relu', scope='tf_layer_8')

        actor_output = self.layer_manager.create_output(layer_8, self.output_size, activation_fn='softmax', scope='tf_actor_output')
        critic_output = self.layer_manager.create_output(layer_8, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])
        return actor_output, critic_output, layer_7, layer_2, layer_5

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        with tf.name_scope('log-of-actor-policy'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            log_policy = tf.log(self.tf_actor_output + 1e-6)
            log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
            log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

        with tf.name_scope('actor-entropy'):
            actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actor_output, log_policy))
            tf.summary.scalar("actor-entropy", actor_entropy)

        with tf.name_scope('actor-loss'):
            actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
            tf.summary.scalar('actor-loss', actor_loss)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_loss + (actor_entropy * self.beta))
            tf.summary.scalar('total-loss', total_loss)

        return critic_diff, log_policy, log_policy_action, actor_entropy, actor_loss, critic_loss, total_loss

    def get_input_shape(self):
        return self.input_shape

    def predict(self, session, state):
        st = state[0]
        map_data = state[1]
        feed_dict = {self.tf_inputs: st, self.tf_map_inputs: map_data}
        return session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

    def get_params(self, data_dict):
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        next_states = data_dict['next_states']
        terminals = data_dict['terminals']
        learning_rate = data_dict['learning_rate']
        logging = data_dict['logging']
        global_step = data_dict['global_step']
        map_data = data_dict['map_data']
        return states, actions, rewards, next_states, terminals, learning_rate, logging, global_step, map_data

    def train(self, session, data_dict):
        states, actions, rewards, next_states, terminals, learning_rate, logging, _, other_data = \
            self.get_params(data_dict)

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1], other_data[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_rewards: td_rewards,
                     self.tf_map_inputs: other_data,
                     self.tf_learning_rate: learning_rate}

        if logging:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                    session.run([self.tf_actor_output, self.tf_critic_output,
                                         self.tf_log_policy, self.tf_log_policy_action,
                                         self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
                                         self.tf_total_loss,
                                         ], feed_dict=feed_dict)
                print("ACTOR OUTPUT:")
                print(actor_output)
                print("CRITIC OUTPUT:")
                print(critic_output)
                print("LOG POLICY:")
                print(log_policy)
                print("LOG POLICY ACTION:")
                print(log_policy_action)
                print("CRITIC DIFF:")
                print(critic_diff)
                print("CRITIC LOSS:")
                print(critic_loss)
                print("ACTOR LOSS:")
                print(actor_loss)
                print("ACTOR ENTROPY:")
                print(actor_entropy)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state, other_data):
        feed_dict = {self.tf_inputs: [state], self.tf_map_inputs: [other_data]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables