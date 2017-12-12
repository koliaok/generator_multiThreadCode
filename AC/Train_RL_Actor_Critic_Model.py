
"""
    Script to solve generation source code problem using policy gradient with neural network
	function approximation
"""

import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import AC.Code_Generation_Agent as agentModel
import AC.Actor_Critic_Model as a3cModel


class ActorCriticLearner:
    def __init__(self, agent, max_episodes, episodes_before_update, max_reward, timstep):
        self.agent = agent
        self.actor = a3cModel.Actor(self.agent)
        self.actor.actor_start_session()

        self.critic = a3cModel.Critic(self.agent)
        self.critic.critic_start_session()


        print("success")

        self.max_episodes = max_episodes
        self.episodes_before_update = episodes_before_update
        self.max_reward = max_reward
        self.timeStep = timstep

    def learn(self):

        advantage_vectors = []
        sum_reward = 0
        update = True
        for i in range(self.max_episodes):
            episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, episode_total_reward = self.actor.rollout_policy(
                self.timeStep, i + 1)
            advantage_vector = self.critic.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
            advantage_vectors.append(advantage_vector)
            sum_reward += episode_total_reward
            if (i + 1) % self.episodes_before_update == 0:
                avg_reward = sum_reward / self.episodes_before_update
                print("Current {} episode average reward: {:.3f}".format(self.episodes_before_update, avg_reward))
                # In this part of the code I try to reduce the effects of randomness leading to oscillations in my
                # network by sticking to a solution if it is close to final solution.
                # If the average reward for past batch of episodes exceeds that for solving the environment, continue with it
                if avg_reward >= self.max_reward:  # This is the criteria for having solved the environment by Open-AI Gym
                    update = False
                else:
                    update = True

                if update:
                    print("Updating")
                    self.actor.update_policy(advantage_vectors)
                    self.critic.update_value_estimate()
                else:
                    print("Good Solution, not updating")
                    self.actor.actor_save_session()
                    self.critic.critic_save_session()
                    break
                # Delete the data collected so far
                del advantage_vectors[:]
                self.actor.reset_memory()
                sum_reward = 0



def main():
    max_episodes = 1000000
    episodes_before_update = 2.0
    max_reward = 400
    timeStep = 150
    agent = agentModel.Agent()

    ac_learner = ActorCriticLearner(agent, max_episodes, episodes_before_update, max_reward,timeStep)
    ac_learner.learn()


if __name__ == "__main__":
    main()