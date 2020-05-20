import tensorflow as tf
import numpy as np

from network import Network


class NetworksManager:
    def __init__(self, config, state_dimension, action_dimension, population_manager):
        self.config = config
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.is_shared_buffer = self.config['evolution']['is_shared_buffer']

        self.population_manager = population_manager

        # generate inputs
        self.state_inputs = tf.compat.v1.placeholder(tf.float32, (None, self.state_dimension), name='state_inputs')
        inputs= self.state_inputs
        self.networks = {
            i: Network(
                self.population_manager.randomize_actor_config(), i, state_dimension, action_dimension,inputs)
            for i in range(self.config['evolution']['population'])
        }

        self.ids = list(self.networks.keys())
        self.scores = [0.0 for _ in self.ids]



    def _arrange_by_ids(self, ordered_collection, ids=None):
        if ids is None:
            ids = self.ids
        return {ids[i]: ordered_collection[i] for i in range(len(ids))}

    def set_scores(self, new_scores):
        assert len(new_scores) == len(self.scores)
        mean = np.mean(list(new_scores.values()))
        std = np.std(list(new_scores.values()))
        for i, id in enumerate(self.ids):
            self.scores[i] = (new_scores[id] - mean) / max((std, 0.000001))

    def softmax_select_ids(self, times_to_select):
        logits = np.array(self.scores) * self.config['evolution']['softmax_temperature']
        logits = logits - np.max(logits)  # remove max for numerical stability
        prob = np.exp(logits)
        prob = prob / np.sum(prob)
        if (self.is_shared_buffer == False): #TODO:check that true
            return list(np.random.choice(self.ids, times_to_select, True, [1,0]))
        return list(np.random.choice(self.ids, times_to_select, True, prob))

    def get_best_scoring_actor_id(self):
        return self.ids[np.argmax(self.scores)]

    def train_critics(self, state_inputs, q_label,one_hot_vector_bprop, sess):
        #for network_id in self.ids:
        #    self.networks[network_id].one_hot_vector = one_hot_vector_bprop
        feed_dictionary = self._generate_feed_dictionary(state_inputs, q_label,one_hot_vector_bprop)
        critic_summaries = [self.networks[network_id].critic_optimization_summaries for network_id in self.ids]
        critic_optimizations = [self.networks[network_id].optimize_critic for network_id in self.ids]
        all_steps = critic_summaries + critic_optimizations
        execution_result = sess.run(all_steps, feed_dictionary)
        summaries_results = execution_result[:self.config['evolution']['population']]
        return self._arrange_by_ids(summaries_results)



    #predicted q values
    def predict_action(self, state_inputs, sess, use_online_network, ids=None):
        feed_dictionary = self._generate_feed_dictionary(state_inputs)
        if ids is None:
            ids = self.ids
        action_prediction = []

        for network_id in ids:
            if use_online_network:
                action_prediction.append(self.networks[network_id].online_q_value)#online_action
            else:
                action_prediction.append(self.networks[network_id].target_q_value)
        action_results = sess.run(action_prediction, feed_dictionary)
        return self._arrange_by_ids(action_results, ids)



    def update_target_networks(self, sess):
        critic_updates = [self.networks[network_id].update_critic_target_params for network_id in self.ids]
        sess.run(critic_updates)

    #TODO: important note: now we send the same one hot to all networks, check if we need a diffrent one for each net
    def _generate_feed_dictionary(self, state_inputs, scalar_inputs=None,one_hot_vector = None):
        feed_dictionary = {self.state_inputs: state_inputs}#, self.one_hot_vector : one_hot_vector}

        if scalar_inputs is not None:
            for network_id in scalar_inputs:
                feed_dictionary[self.networks[network_id].scalar_label] = scalar_inputs[network_id]
                feed_dictionary[self.networks[network_id].one_hot_vector] = one_hot_vector


            #for network_id in self.ids:
                #self.networks[network_id].one_hot_vector = one_hot_vector
        return feed_dictionary
