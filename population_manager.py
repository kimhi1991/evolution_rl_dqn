import copy
import numpy as np


class PopulationManager:

    def __init__(self, config, base_agent_config):
        self.config = config
        self.base_agent_config = base_agent_config

    def randomize_actor_config(self):
        if self.config['evolution']['use_default_configration']:
            return self.base_agent_config
        return self._change_config(copy.deepcopy(self.base_agent_config))

    def _change_config(self, config_to_morph):
        config_to_morph['model']['tau'] = self._select_uniform_log_scale(0.005, 0.1)

        config_to_morph['actor']['learning_rate'] = self._select_uniform_log_scale(0.00001, 0.1)
        config_to_morph['actor']['gradient_limit'] = self._select_uniform_log_scale(0.001, 1.0,
                                                                                    zero_value_probability=0.1)
        config_to_morph['actor']['tanh_preactivation_loss_coefficient'] = self._select_uniform_log_scale(
            0.001, 10.0, zero_value_probability=0.1)

        config_to_morph['critic']['learning_rate'] = self._select_uniform_log_scale(0.00001, 0.1)
        config_to_morph['critic']['gradient_limit'] = self._select_uniform_log_scale(0.001, 1.0,
                                                                                     zero_value_probability=0.1)
        config_to_morph['critic']['l2_regularization_coefficient'] = self._select_uniform_log_scale(
            0.0000001, 0.001, zero_value_probability=0.1)
        return config_to_morph

    @staticmethod
    def _select_uniform_log_scale(minimal, maximal, zero_value_probability=None):
        if zero_value_probability is not None:
            zero_selected = np.random.binomial(1, zero_value_probability)
            if zero_selected:
                return 0.0
        return float(np.exp(np.random.uniform(np.log(minimal), np.log(maximal))))