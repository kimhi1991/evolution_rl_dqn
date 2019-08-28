import os
import random
import datetime
import tensorflow as tf
import yaml
import time
import numpy as np



from envs import EnvGenerator
from episode_runner import EpisodeRunner
from networks_manager import NetworksManager
from population_manager import PopulationManager
from replay_buffer import ReplayBuffer
from summaries_collector import SummariesCollector


def run_for_config(config, agent_config, env_generator, is_in_collab=False):
    # set the name of the model
    model_name = config['general']['name']
    now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    model_name = now + '_' + model_name if model_name is not None else now

    #random seed
    random_seed = config['general']['random_seed']
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.compat.v1.set_random_seed

    # where we save all the outputs
    working_dir = os.getcwd()
    if is_in_collab:
        working_dir = '/' + os.path.join('content', 'gdrive', 'My Drive', 'colab_data', 'EvoDDPG')
    saver_dir = os.path.join(working_dir, 'models', model_name)
    if not os.path.exists(saver_dir):
        os.makedirs(saver_dir)
    config_copy_path = os.path.join(working_dir, 'models', model_name, 'config.yml')
    summaries_dir = os.path.join(working_dir, 'tensorboard', model_name)

    #get enviroment constants
    state_dimension, action_dimension = env_generator.get_env_definitions()

    #constract population manager
    population_manager = PopulationManager(config, agent_config)

    # generate networks
    network_manager = NetworksManager(config, state_dimension, action_dimension, population_manager)

    # initialize replay memory
    replay_buffer = ReplayBuffer(config)

    # save model
    saver = tf.train.Saver(max_to_keep=4, save_relative_paths=saver_dir)
    yaml.dump(config, open(config_copy_path, 'w'))
    summaries_collector = SummariesCollector(summaries_dir, model_name)
    episode_runner = EpisodeRunner(config, env_generator.get_env_wrapper().get_env(), network_manager)
    visualization_episode_runner = EpisodeRunner(
        config, env_generator.get_env_wrapper().get_env(), network_manager, is_in_collab=is_in_collab)

    test_results = []


    def update_model(sess):
        batch_size = config['model']['batch_size']
        gamma = config['model']['gamma']
        current_state, action, reward, terminated, next_state = replay_buffer.sample_batch(batch_size)


        # get the predicted q value of the next state (action is taken from the target policy)
        #network_manager.runing_network == 1 - network_manager.runing_network
        next_state_action_target_q = network_manager.predict_policy_q(next_state, sess, use_online_network=False)

        # compute critic label
        q_label = {}

        one_hot_vector = []
        for i in range(batch_size):
            c= np.zeros(4)
            c[action[i]]=1
            one_hot_vector.append(c)

        reward_batch=[]
        for i in range(batch_size):
            d = np.zeros(4)
            d[action[i]] = reward[i]
            reward_batch =np.concatenate((reward_batch,d),axis=0)
        reward_batch=np.reshape(reward_batch,( batch_size,-1))





        terminated = [terminated,terminated,terminated,terminated]
        terminated= np.transpose(terminated)

        for network_id in next_state_action_target_q:
            q_label[network_id] = \
                np.expand_dims(np.array(reward_batch) +
                               np.multiply(
                                   np.multiply(1 - np.array(terminated), gamma),
                                   np.array(next_state_action_target_q[network_id])), 1)

        for network_id in next_state_action_target_q:

            q_label[network_id] = np.multiply(np.squeeze(np.array(q_label[network_id])) , np.array(one_hot_vector))


        # train critic given the targets
        critic_optimization_summaries = network_manager.train_critics(current_state, q_label, sess)


        # update target networks
        # close because of double dqn
        network_manager.update_target_networks(sess)
        result = list(critic_optimization_summaries.values()) #+ list(actor_optimization_summaries.values())
        return result

    def compute_actor_score(episode_rewards, episode_lengths):
        return sum(episode_rewards)

    with tf.Session(
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
            )
    ) as sess:
        sess.run(tf.global_variables_initializer())
        #close because of double dqn
        network_manager.update_target_networks(sess)

        global_step = 0
        total_episodes = 0
        for update_index in range(config['general']['updates_cycle_count']):

            episode_rewards, episode_lengths = [], []
            episodes_per_update = config['general']['episodes_per_update']
            actor_ids = network_manager.softmax_select_ids(episodes_per_update)
            total_rollout_time = None
            for actor_id in actor_ids:
                # run episode:
                states, actions, rewards, done, rollout_time = episode_runner.run_episode(sess, actor_id, True)
                if total_rollout_time is None:
                    total_rollout_time = rollout_time
                else:
                    total_rollout_time += rollout_time
                # at the end of episode
                replay_buffer.add_episode(states, actions, rewards, done)
                total_episodes += 1

                episode_rewards.append(sum(rewards))
                episode_lengths.append(len(rewards))

            print( 'rollout time took: {}'.format(total_rollout_time))
            print(rewards[-1])

            # do updates
            if replay_buffer.size() > config['model']['batch_size']:
                a = datetime.datetime.now()
                for _ in range(config['general']['model_updates_per_cycle']):
                    summaries = update_model(sess)
                    if global_step % config['general']['write_train_summaries'] == 0:
                        summaries_collector.write_train_episode_summaries(
                            sess, global_step, episode_rewards, episode_lengths
                        )
                        summaries_collector.write_train_optimization_summaries(summaries, global_step)
                    global_step += 1
                b = datetime.datetime.now()
                #print 'update took: {}'.format(b - a)

            # test if needed
            if update_index % config['test']['test_every_cycles'] == 0:
                # run test
                number_of_episodes_per_actor = config['test']['number_of_episodes_per_actor']
                actor_scores = {}
                actor_stats = {}
                for actor_id in network_manager.ids:
                    episode_rewards, episode_lengths = [], []
                    for _ in range(number_of_episodes_per_actor):
                        states, actions, rewards, done, rollout_time = episode_runner.run_episode(sess, actor_id, False)
                        # at the end of episode
                        episode_rewards.append(sum(rewards))
                        episode_lengths.append(len(rewards))
                    actor_scores[actor_id] = compute_actor_score(episode_rewards, episode_lengths)
                    actor_stats[actor_id] = (episode_rewards, episode_lengths)
                # update the scores
                network_manager.set_scores(actor_scores)
                # get the statistics of the best actor:
                best_actor_id = network_manager.get_best_scoring_actor_id()
                episode_rewards, episode_lengths = actor_stats[best_actor_id]
                summaries_collector.write_test_episode_summaries(sess, global_step, episode_rewards, episode_lengths)
                # run visualization with the best actor
                #--------------need to change when we have population----------------------
                visualization_episode_runner.run_episode(sess, best_actor_id, False, render=config['test']['show_best'])

            if update_index % config['general']['save_model_every_cycles'] == 0:
                saver.save(sess, os.path.join(saver_dir, 'all_graph'), global_step=global_step)
    return test_results


if __name__ == '__main__':
    # disable tf warning
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # read the config
    with open(os.path.join(os.getcwd(), 'config/config.yml'), 'r') as yml_file:
        config = yaml.load(yml_file)
        print('------------ config ------------')
        print(yaml.dump(config))

    # read the agent config
    with open(os.path.join(os.getcwd(), 'config/agent.yml'), 'r') as yml_file:
        agent_config = yaml.load(yml_file)
        print('------------ agent ------------')
        print(yaml.dump(config))

    run_for_config(config, agent_config, EnvGenerator(config['general']['gym_continuous_env']))
