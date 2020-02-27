import datetime
import time
import os
import tensorflow as tf
import numpy as np


class SummariesCollector:
    def __init__(self, summaries_dir, model_name,config):
        self.config= config
        completion_reward = self.config['general']['completion_reward']
        self._train_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train_' + model_name))
        self.write_train_episode_summaries = self._init_episode_summaries('train', self._train_summary_writer,completion_reward)

        self._test_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'test_' + model_name))
        self.write_test_episode_summaries = self._init_episode_summaries('test', self._test_summary_writer,completion_reward)

    @staticmethod
    def _init_episode_summaries(prefix, summary_writer,completion_reward):

        min_episode_reward_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        mean_episode_reward_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        max_episode_reward_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        min_episode_length_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        mean_episode_length_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        max_episode_length_var = tf.Variable(0, trainable=False, dtype=tf.float32)

        #if (prefix == 'test'):
        #    if(mean_episode_reward_var > completion_reward):
             #    print ("avg test is greater then complition reward, the agent won the game!!")
             #   return
        summaries = tf.summary.merge([
            tf.summary.scalar(prefix + '_min_episode_reward_var', min_episode_reward_var),
            tf.summary.scalar(prefix + '_mean_episode_reward_var', mean_episode_reward_var),
            tf.summary.scalar(prefix + '_max_episode_reward_var', max_episode_reward_var),
            tf.summary.scalar(prefix + '_min_episode_length_var', min_episode_length_var),
            tf.summary.scalar(prefix + '_mean_episode_length_var', mean_episode_length_var),
            tf.summary.scalar(prefix + '_max_episode_length_var', max_episode_length_var),
        ])

        def write_episode_summaries(sess, global_step, episode_rewards, episode_lengths,actor_id):
            if (prefix == 'test' and np.mean(episode_rewards) > completion_reward):
                   print('{} network {} - {}: episode mean reward:  , {} agent won the game'.format(
                           datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),actor_id, prefix,
                            np.mean(episode_rewards)))
            else:
                print( '{} network {} - {}: episode reward: min, mean, max {}, {}, {} episode length min, mean, max {}, {}, {}'.format(
                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),actor_id ,prefix,
                    np.min(episode_rewards), np.mean(episode_rewards), np.max(episode_rewards), np.min(episode_lengths),
                    np.mean(episode_lengths), np.max(episode_lengths)
                ))

            summary_str = sess.run(summaries, feed_dict={
                min_episode_reward_var: np.min(episode_rewards),
                mean_episode_reward_var: np.mean(episode_rewards),
                max_episode_reward_var: np.max(episode_rewards),
                min_episode_length_var: np.min(episode_lengths),
                mean_episode_length_var: np.mean(episode_lengths),
                max_episode_length_var: np.max(episode_lengths),
            })


            summary_writer.add_summary(summary_str, global_step)
            summary_writer.flush()

        return write_episode_summaries

    def write_train_optimization_summaries(self, summaries, global_step):
        for s in summaries:
            if s is not None:
                self._train_summary_writer.add_summary(s, global_step)
        self._train_summary_writer.flush()



