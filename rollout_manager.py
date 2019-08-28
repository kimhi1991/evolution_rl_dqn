import time
import os
import copy

import numpy as np
import tensorflow as tf
import multiprocessing
import Queue
import datetime

from network import Network


class RolloutManager:
    def __init__(self, config, env_generator, actor_processes=None, fixed_queries=None):
        self.episode_generation_queue = multiprocessing.JoinableQueue()
        self.episode_results_queue = multiprocessing.Queue()
        if actor_processes is None:
            actor_processes = config['general']['actor_processes']
        self.fixed_queries = fixed_queries

        self.actor_specific_queues = [multiprocessing.JoinableQueue() for _ in range(actor_processes)]

        self.actors = [
            ActorProcess(
                copy.deepcopy(config), self.episode_generation_queue, self.episode_results_queue,
                self.actor_specific_queues[i], env_generator
            )
            for i in range(actor_processes)
        ]

        # start all the actor processes
        for a in self.actors:
            a.start()
        # for every actor process, post a message to initialize the actor network
        for actor_queue in self.actor_specific_queues:
            actor_queue.put((0, ))
            actor_queue.join()

    def generate_episodes(self, number_of_episodes, is_train):
        for i in range(number_of_episodes):
            # place in queue
            self.episode_generation_queue.put(is_train)

        self.episode_generation_queue.join()

        episodes = []
        while number_of_episodes:
            number_of_episodes -= 1
            episodes.append(self.episode_results_queue.get())

        return episodes

    def set_policy_weights(self, weights):
        message = (2, weights)
        self._post_private_message(message, self.actor_specific_queues)

    def end(self):
        message = (1, )
        self._post_private_message(message, self.actor_specific_queues)

    @staticmethod
    def _post_private_message(message, queues):
        for queue in queues:
            queue.put(message)
        for queue in queues:
            queue.join()
