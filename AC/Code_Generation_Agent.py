from __future__ import print_function
import tensorflow as tf

import argparse
import os
import numpy as np
from six.moves import cPickle
from AC.RnnCodeGenModel import Model

#Define Agent Moder and Enviroment
class Agent:
    def __init__(self):

        parser = argparse.ArgumentParser(
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--save_dir', type=str, default='../save/',
                            help='model directory to store checkpointed models')
        parser.add_argument('--data_dir', type=str, default='../data/sourceCODE',
                            help='data directory containing input.txt')
        parser.add_argument('-n', type=int, default=1000,
                            help='number of characters to sample')
        parser.add_argument('--sample', type=int, default=1,
                            help='0 to use max at each timestep, 1 to sample at '
                                 'each timestep, 2 to sample on spaces')

        self.args = parser.parse_args()

        with open(os.path.join(self.args.save_dir, 'config.pkl'), 'rb') as f:
            self.saved_args = cPickle.load(f)
            self.model = Model(self.saved_args, self.args, training=False)

        with open(os.path.join(self.args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
            self.chars, self.vocab = cPickle.load(f)
        with open(os.path.join(self.args.data_dir, 'data.npy'), 'rb') as f:
            self.tensor = np.load(f)
        self.init_sequenceSize = 2
        self.limit = 170000


    def observation_space(self):
        return self.init_sequenceSize

    def action_space(self):
        return self.limit

    def reset(self):
        self.model.startSession_Agent()
        code, state = self.model.getState( self.chars, self.vocab, self.init_sequenceSize, self.limit,
                                          self.args.sample)  # 학습된 모델을 바탕으로 단어를 생성하여 받아옴
        print(code)
        return state
    def step(self, action):
        code, next_state, reward = self.model.get_step( self.chars, self.vocab, action,self.init_sequenceSize, self.limit, self.args.sample)
        done = False
        return next_state, reward, done

    def single_word_step(self, action):
        code, next_state, reward = self.model.single_get_step( self.chars, self.vocab, action,
                                                   self.init_sequenceSize, self.limit, self.args.sample)
        print(code)
        done = False

        return next_state, reward, done

#thread = 347
#Runnable= 374
#extends = 70
#public = 11
#class = 43
#11 = public
#66366 = typeProcessor
#161648 = SMPTE_24
#84120 = ForeignSession
#47738 = fGlobalGroupDecls
#68272 = Tethering
#61323 = xsdFileName
#113734 = JUnitLaunchConfigurationConstants
"""
def main():
    agent = Agent()
    agent.model.startSession_Agent()
    state = agent.reset()
    for n in range(0,100):
        if n < 60:
            action = 11
        else:
            action = 43

        next_state, reward, done = agent.single_word_step(action=action)
        print("next_state: ", next_state)
        print(len(next_state))
        print("reward: ", reward)

if __name__ == '__main__':
    main()
"""