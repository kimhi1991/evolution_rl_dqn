import os
import tensorflow as tf
import tensorflow.contrib.layers as layers


class Network(object):
    def __init__(self, config, id, state_dimension, action_dimension, inputs):
        self.config = config
        self.id = id

        # input related data
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        #self.one_hot_vector = one_hot_vector
        self.state_inputs = inputs
        #self.action_inputs = global_inputs[1]

        with tf.compat.v1.variable_scope('network_{}'.format(id)):
            self.scalar_label = tf.compat.v1.placeholder(tf.float32, [None, 4], name='scalar_input')
            # since we take partial derivatives w.r.t subsets of the parameters, we always need to remember which
            # parameters are currently being added. note that this also causes the model to be non thread safe,
            # therefore the creation must happen sequentially
            variable_count = len(tf.compat.v1.trainable_variables())
            tau = self.config['agent']['tau']
            # import pdb; pdb.set_trace()




            #online network 1
            self.online_q_value = self._create_critic_network(
                self.state_inputs, is_online=True, reuse_flag=False, add_regularization_loss=False
            )

            online_critic_params = tf.trainable_variables()[variable_count:]
            variable_count = len(tf.compat.v1.trainable_variables())

            # target network 1
            # predicting the q value to avoid over astimating
            self.target_q_value = self._create_critic_network(
                self.state_inputs, is_online=False, reuse_flag=False, add_regularization_loss=False
            )



            assert variable_count == len(tf.trainable_variables()[variable_count:])  # make sure no new parameters were added
            target_critic_params = tf.trainable_variables()[variable_count:]



            # periodically update target critic with online critic weights
            self.update_critic_target_params = [target_critic_params[i].assign(
                    tf.multiply(online_critic_params[i], tau) + tf.multiply(target_critic_params[i], 1. - tau)
                ) for i in range(len(target_critic_params))]




            batch_size = tf.cast(tf.shape(self.state_inputs)[0], tf.float32)

            # critic optimization
            #we switched the online_q_value_fixed to not fixed
            critic_prediction_loss = tf.div(
                tf.losses.mean_squared_error(self.scalar_label, self.online_q_value), batch_size)
            critic_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            critic_regularization_loss = tf.div(tf.add_n(critic_regularization), batch_size) \
                if len(critic_regularization) > 0 else 0.0
            self.critic_total_loss = critic_prediction_loss + critic_regularization_loss

            self.critic_initial_gradients_norm, self.critic_clipped_gradients_norm, self.optimize_critic = \
                self._optimize_by_loss(
                    self.critic_total_loss, online_critic_params, self.config['critic']['learning_rate'],
                    self.config['critic']['gradient_limit']
                )

            # summaries for the critic optimization
            self.critic_optimization_summaries = tf.summary.merge([
                #tf.summary.scalar('critic_regularization_loss', critic_regularization_loss),
                tf.summary.scalar('critic_total_loss', self.critic_total_loss),
                #tf.summary.scalar('critic_gradients_norm_initial', self.critic_initial_gradients_norm),
                #tf.summary.scalar('critic_gradients_norm_clipped', self.critic_clipped_gradients_norm),
                #tf.summary.scalar('critic_mean_prediction', tf.reduce_mean(self.online_q_value_fixed_action)),
                #tf.summary.histogram('critic_prediction_distribution', self.online_q_value_fixed_action),
            ])

    @staticmethod
    def _optimize_by_loss(loss, parameters_to_optimize, learning_rate, gradient_limit):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss, parameters_to_optimize))
        initial_gradients_norm = tf.global_norm(gradients)
        if gradient_limit > 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_limit, use_norm=initial_gradients_norm)
        clipped_gradients_norm = tf.global_norm(gradients)
        optimize_op = optimizer.apply_gradients(zip(gradients, variables))
        return initial_gradients_norm, clipped_gradients_norm, optimize_op

    @staticmethod
    def _get_activation(activation):
        if activation == 'relu':
            return tf.nn.relu
        if activation == 'tanh':
            return tf.nn.tanh
        if activation == 'elu':
            return tf.nn.elu
        return None


    def _create_critic_network(self, state_inputs, is_online, reuse_flag, add_regularization_loss):
        name_prefix = '{}_critic_{}'.format(os.getpid(), 'online' if is_online else 'target')
        layers_before_action = self.config['critic']['layers_before_action']
        #layers_after_action = self.config['critic']['layers_after_action']
        activation = self._get_activation(self.config['critic']['activation'])

        current = state_inputs
        # import pdb; pdb.set_trace()
        scale = self.config['critic']['l2_regularization_coefficient'] if add_regularization_loss else 0.0

        for i, layer_size in enumerate(layers_before_action):
            #pdb.set_trace()
            current = tf.layers.dense(
                current, layer_size, activation=activation, name='{}_before_action_{}'.format(name_prefix, i),
                reuse=reuse_flag, kernel_regularizer=layers.l2_regularizer(scale)
            )

        #current = tf.concat((current, self.one_hot_vector), axis=1)

        if self.config['critic']['last_layer_tanh']:
            q_val = tf.layers.dense(
                current, 4, activation=tf.nn.tanh, name='{}_tanh_layer'.format(name_prefix), reuse=reuse_flag,
                kernel_regularizer=layers.l2_regularizer(scale)
            )

            q_val_with_stretch = tf.layers.dense(
                tf.ones_like(q_val), 4, tf.abs, False, name='{}_stretch'.format(name_prefix), reuse=reuse_flag,
                kernel_regularizer=layers.l2_regularizer(scale)
            ) * q_val
            return q_val_with_stretch
        else:
            q_val = tf.layers.dense(
                current, 4, activation=None, name='{}_linear_layer'.format(name_prefix), reuse=reuse_flag,
                kernel_regularizer=layers.l2_regularizer(scale)
            )
            # pdb.set_trace()
            return q_val


    def get_actor_online_weights(self, sess):
        return sess.run(self.online_actor_params)

    def set_actor_online_weights(self, sess, weights):
        feed = {
            self.online_actor_parameter_weights_placeholders[var.name]: weights[i]
            for i, var in enumerate(self.online_actor_params)
        }
        sess.run(self.online_actor_parameters_assign_ops, feed)



    # def _print(self, header, array):
    #     print header
    #     print 'is nan? {}'.format(np.isnan(array).any())
    #     print 'max {}'.format(np.max(array))
    #     print 'min {}'.format(np.min(array))
    #     print ''
    #
    # def debug_all(self, state_inputs, action, q_label, sess):
    #     feed_dictionary = self._generate_feed_dictionary(state_inputs, action)
    #     feed_dictionary[self.scalar_label] = q_label
    #     ops = [
    #         self.online_action, self.target_action, self.online_q_value_fixed_action, self.online_q_value,
    #         self.target_q_value, self.critic_total_loss, self.critic_initial_gradients_norm,
    #         self.critic_clipped_gradients_norm, self.actor_loss, self.actor_initial_gradients_norm,
    #         self.actor_clipped_gradients_norm
    #     ]
    #     all_steps = sess.run(ops, feed_dictionary)
    #     self._print('self.online_action', all_steps[0])
    #     self._print('self.target_action', all_steps[1])
    #     self._print('self.online_q_value_fixed_action', all_steps[2])
    #     self._print('self.online_q_value', all_steps[3])
    #     self._print('self.target_q_value', all_steps[4])
    #     self._print('self.critic_total_loss', all_steps[5])
    #     self._print('self.critic_initial_gradients_norm', all_steps[6])
    #     self._print('self.critic_clipped_gradients_norm', all_steps[7])
    #     self._print('self.actor_loss', all_steps[8])
    #     self._print('self.actor_initial_gradients_norm', all_steps[9])
    #     self._print('self.actor_clipped_gradients_norm', all_steps[10])


