import os
import tensorflow as tf
import tensorflow.contrib.layers as layers
import  numpy as np


#TODO: initializing oh when create net
class Network(object):
    def __init__(self, config, id, state_dimension, action_dimension, inputs):



        self.config = config

        self.id = id
        # input related data
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension

        self.state_inputs = inputs
        #self.action_inputs = global_inputs[1]

        with tf.compat.v1.variable_scope('network_{}'.format(id)):
            self.scalar_label = tf.compat.v1.placeholder(tf.float32, [None, 1], name='scalar_input')
            # since we take partial derivatives w.r.t subsets of the parameters, we always need to remember which
            # parameters are currently being added. note that this also causes the model to be non thread safe,
            # therefore the creation must happen sequentially
            self.one_hot_vector = tf.compat.v1.placeholder(tf.float32, [None, self.action_dimension], name='one_hot_vector') #check if the name important
            variable_count = len(tf.compat.v1.trainable_variables())
            tau = self.config['agent']['tau']


            #online network 1
            self.online_q_value = self._create_critic_network(
                self.state_inputs, is_online=True, reuse_flag=False, add_regularization_loss=False
            )

            online_critic_params = tf.trainable_variables()[variable_count:]
            self.online_critic_params =online_critic_params
            variable_count = len(tf.compat.v1.trainable_variables())

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
            #TODO: dot oh vec (chosen by actions) -- tf.reduce_sum, tf.multiply

            #q_val_one_hot = tf.matmul(self.online_q_value, self.one_hot_vector)
            #q_val_one_hot = tf.reduce_sum(tf.multiply(self.online_q_value, self.one_hot_vector),1)
            #q_val_one_hot=tf.expand_dims(q_val_one_hot, 1)
            q_val_one_hot = tf.expand_dims(tf.reduce_sum(self.online_q_value* self.one_hot_vector,axis=1), axis=-1)


            critic_prediction_loss = tf.div(
                tf.losses.mean_squared_error(self.scalar_label, q_val_one_hot), batch_size)
                #tf.losses.mean_squared_error(self.scalar_label, self.online_q_value), batch_size)
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
                tf.summary.scalar('critic_total_loss', self.critic_total_loss),
            ])


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
        if self.config['critic']['last_layer_tanh']:
            q_val = tf.layers.dense(
                current, self.action_dimension, activation=tf.nn.tanh, name='{}_tanh_layer'.format(name_prefix), reuse=reuse_flag,
                kernel_regularizer=layers.l2_regularizer(scale)
            )

            q_val_with_stretch = tf.layers.dense(
                tf.ones_like(q_val), self.action_dimension, tf.abs, False, name='{}_stretch'.format(name_prefix), reuse=reuse_flag,
                kernel_regularizer=layers.l2_regularizer(scale)
            ) * q_val
            return q_val_with_stretch

        else:
            #activation=None to softmax
            q_val = tf.layers.dense(
                current, self.action_dimension, activation=None, name='{}_linear_layer'.format(name_prefix), reuse=reuse_flag,
                kernel_regularizer=layers.l2_regularizer(scale)
            )
            return q_val


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



    def get_critic_online_weights(self, sess):
        return sess.run(self.online_critic_params)

    def set_actor_online_weights(self, sess, weights):
        feed = {
            self.online_actor_parameter_weights_placeholders[var.name]: weights[i]
            for i, var in enumerate(self.online_actor_params)
        }
        sess.run(self.online_actor_parameters_assign_ops, feed)
