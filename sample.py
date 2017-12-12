from __future__ import print_function
import tensorflow as tf

import argparse
import os
import numpy as np
from six.moves import cPickle

from model import Model

from six import text_type



def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('--data_dir', type=str, default='data/sourceCODE', ## 데이터 저장 장소
                        help='data directory containing input.txt')
    parser.add_argument('-n', type=int, default=100,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    with open(os.path.join(args.data_dir, 'data.npy'), 'rb') as f:
        tensor = np.load(f)

    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            longWords, vector_list = model.sample(sess, chars, vocab,tensor,args.n, args.sample)#학습된 모델을 바탕으로 단어를 생성하여 받아옴

            print(longWords)
            print(vector_list)


if __name__ == '__main__':
    main()
