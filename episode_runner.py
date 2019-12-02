import time

import numpy as np
import datetime
from random import randrange
import os
import yaml

with open(os.path.join(os.getcwd(), 'config/config.yml'), 'r') as yml_file:
    config = yaml.load(yml_file)


class EpisodeRunner:
    def __init__(self, config, gym_env, networks_manager, is_in_collab=False):
        self.config = config
        self.gym_env = gym_env
        self.networks_manager = networks_manager
        self.epsilon = self.config['model']['epsilon']
        #self.time =0
        self.is_in_collab = is_in_collab
        if self.is_in_collab:
            from pyvirtualdisplay import Display
            display = Display(visible=0, size=(400, 300))
            display.start()


    def _render(self, render):
        if render:
            if self.is_in_collab:
                import matplotlib.pyplot as plt
                from IPython import display as ipythondisplay
                screen = self.gym_env.render(mode='rgb_array')

                plt.imshow(screen)
                ipythondisplay.clear_output(wait=True)
                ipythondisplay.display(plt.gcf())
            else:
                self.gym_env.render()
                time.sleep(0.01)

    def run_episode(self, sess, actor_id, is_train, render=False):
        # the trajectory data structures to return
        states = [self.gym_env.reset()]
        self._render(render)
        actions = []
        rewards = []
        done = False

        # set the start state
        start_rollout_time = datetime.datetime.now()
        max_steps = self.config['general']['max_steps']

        #exploration in train
        if (is_train == True):
            if (self.epsilon >= self.config['model']['epsilon_min']):  # and self.time > 10):
                self.epsilon *= self.config['model']['epsilon_decay']
            else:
                self.epsilon = self.config['model']['epsilon_min']
        #print(self.epsilon)
        for j in range(max_steps):
            #choose the best action w.r.t q-value
            predicted_actions = self.networks_manager.predict_action(
               [states[-1]], sess, use_online_network=True, ids=[actor_id])[actor_id]

            prob_exploration = np.random.binomial(1, self.epsilon, 1)[0]
            if (self.epsilon  > 0. and np.random.uniform()< self.epsilon and is_train == True):
                chosen_action = np.random.choice(config['general']['action_dim'])
            else:
                chosen_action = np.argmax(predicted_actions)

            next_state, reward, done, _ = self.gym_env.step(chosen_action)
            next_state = np.squeeze(next_state)
            # set a new current state
            states.append(next_state)
            actions.append(chosen_action)
            rewards.append(reward)
            # visualize if needed
            self._render(render)
            # break if needed
            if done:
                break
        # return the trajectory along with query info
        assert len(states) == len(actions) + 1
        assert len(states) == len(rewards) + 1
        end_episode_time = datetime.datetime.now()
        rollout_time = end_episode_time-start_rollout_time
        return states, actions, rewards, done, rollout_time
