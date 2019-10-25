import tensorflow as tf
from fruit.configs.a3c import A3CConfig


class MAA3CConfig(A3CConfig):
    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, beta=0.01,
                 global_norm_clipping=40, debug_mode=False, gamma=0.99, optimizer=None):
        super().__init__(environment=environment, initial_learning_rate=initial_learning_rate,
                         history_length=history_length, beta=beta, global_norm_clipping=global_norm_clipping,
                         debug_mode=debug_mode, gamma=gamma, optimizer=optimizer)

        self.num_of_agents = environment.get_number_of_agents()

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.get_shape())
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions_list, self.tf_rewards, self.tf_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actors_list, self.tf_critic_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        actions_list = []
        for i in range(self.num_of_agents):
            actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions_" + str(i))
            actions_list.append(actions)

        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")

        return inputs, actions_list, rewards, inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_2')

        layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu',
                                                                  scope='tf_layer_3')

        critic_output = self.layer_manager.create_output(layer_3, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])

        actors_list = []
        for i in range(self.num_of_agents):
            actors_list.append(self.layer_manager.create_output(layer_3, self.output_size, activation_fn='softmax',
                                                                scope='tf_actor_output_' + str(i)))

        return actors_list, critic_output

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

        actor_losses = None
        for i in range(self.num_of_agents):
            with tf.name_scope('log-of-actor-policy'):
                actions_one_hot = tf.one_hot(self.tf_actions_list[i], depth=self.output_size, name='one-hot-' + str(i),
                                             on_value=1.0, off_value=0.0, dtype=tf.float32)
                log_policy = tf.log(self.tf_actors_list[i] + 1e-6)
                log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
                log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

            with tf.name_scope('actor-entropy'):
                actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actors_list[i], log_policy))
                tf.summary.scalar("actor-entropy-" + str(i), actor_entropy)

            with tf.name_scope('actor-loss'):
                actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
                tf.summary.scalar('actor-loss-' + str(i), actor_loss)

            with tf.name_scope('actor-losses'):
                if actor_losses is None:
                    actor_losses = actor_loss + (actor_entropy * self.beta)
                else:
                    actor_losses = actor_losses + actor_loss + (actor_entropy * self.beta)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_losses)
            tf.summary.scalar('total-loss', total_loss)

        return total_loss

    def get_input_shape(self):
        return self.input_shape

    def predict(self, session, state):
        feed_dict = {self.tf_inputs: state}
        return session.run(self.tf_actors_list, feed_dict=feed_dict)

    def train(self, session, data_dict):
        states, actions, rewards, next_states, terminals, learning_rate, logging, _ = self.get_params(data_dict)

        actions_ = []
        for i in range(self.num_of_agents):
            acs = []
            for j in range(len(actions)):
                acs.append(actions[j][i])
            actions_.append(acs)

        actions = actions_

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states,
                     self.tf_rewards: td_rewards,
                     self.tf_learning_rate: learning_rate}

        for i in range(self.num_of_agents):
            feed_dict[self.tf_actions_list[i]] = actions[i]

        session.run([self.tf_train_step], feed_dict=feed_dict)

        if logging:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                total_loss = session.run(self.tf_total_loss, feed_dict=feed_dict)
                print("TOTAL LOSS")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state):
        feed_dict = {self.tf_inputs: [state]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables