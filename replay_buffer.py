from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, config):
        self.buffer_size = config['model']['buffer_size']
        self.count = 0
        self.buffer = deque()

    def add_episode(self, states, actions, rewards, terminated):
        assert len(states) == len(actions) + 1
        assert len(states) == len(rewards) + 1
        for i in range(len(states)-1):
            is_terminated = (i == len(states)-2) and terminated
            self.add(states[i], actions[i], rewards[i], is_terminated, states[i+1])

    def add(self, current_state, action, reward, terminated, next_state):
        experience = (current_state, action, reward, terminated, next_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        count = min([batch_size, self.count])
        batch = random.sample(self.buffer, count)
        return zip(*batch)
