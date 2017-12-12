import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, allArgs,training=True):
        self.args = args
        self.allArgs = allArgs
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []

        self.graph = tf.Graph()
        with self.graph.as_default():

            with tf.variable_scope('initSetting'):
                for _ in range(args.num_layers):
                    cell = cell_fn(args.rnn_size)
                    if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                        cell = rnn.DropoutWrapper(cell,
                                                  input_keep_prob=args.input_keep_prob,
                                                  output_keep_prob=args.output_keep_prob)
                    cells.append(cell)


                self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

                self.input_data = tf.placeholder(
                    tf.int32, [args.batch_size, args.seq_length])
                self.targets = tf.placeholder(
                    tf.int32, [args.batch_size, args.seq_length])

                self.initial_state = cell.zero_state(args.batch_size, tf.float32)##초기 state 구성

            with tf.variable_scope('rnnlm'):
                softmax_w = tf.get_variable("softmax_w",
                                            [args.rnn_size, args.vocab_size],initializer= tf.contrib.layers.xavier_initializer())
                softmax_b = tf.get_variable("softmax_b", [args.vocab_size], initializer= tf.contrib.layers.xavier_initializer())

            embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size],initializer= tf.contrib.layers.xavier_initializer())#shape(65,128)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

            # dropout beta testing: double check which one should affect next line
            if training and args.output_keep_prob:
                inputs = tf.nn.dropout(inputs, args.output_keep_prob)

            inputs = tf.split(inputs, args.seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

            def loop(prev, _):
                prev = tf.matmul(prev, softmax_w) + softmax_b
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return tf.nn.embedding_lookup(embedding, prev_symbol)

            outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
            output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

            self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)
            loss = legacy_seq2seq.sequence_loss_by_example(
                    [self.logits],
                    [tf.reshape(self.targets, [-1])],
                    [tf.ones([args.batch_size * args.seq_length])])
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

            with tf.name_scope('cost'):
                self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
            self.final_state = last_state
            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                    args.grad_clip)
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            # instrument tensorboard
            tf.summary.histogram('logits', self.logits)
            tf.summary.histogram('loss', loss)
            tf.summary.scalar('train_loss', self.cost)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables())
            self.ckpt = tf.train.get_checkpoint_state(self.allArgs.save_dir)



    def startSession_Agent(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            print("Agent Model restor complete")

    def getState(self, chars, vocab, num ,limit,sampling_type=1):## 학습된 모델에서 단어를 생성하는 부분
        self.sentence_start_token = "CODE_START"
        self.sentence_end_token = "CODE_END"
        state_list=[]
        self.limit = limit
        self.reward_dict = {'public':0.4,'class':0.4,'extends':0.6,'Thread':1.2,'@':0.5,'Override':0.5, 'Implements':0.6,
                            'Runnable': 1.2,'start':0.8,'Java':0.4,'lang':0.3,'long':0.3, 'void':0.5, 'private':0.3, 'protected':0.3}

        ret = ""
        char = self.sentence_start_token
        temp_state = []
        check_action = False
        cnt =0
        brace_count = 0
        checkBrace = False
        for n in range(num):
            if checkBrace ==False:
                x = np.zeros((1, 1))
                x[0, 0] = vocab[char]
                feed = {self.input_data: x}
                [probs, state] = self.sess.run([self.probs, self.final_state], feed )
                p = probs[0]

                sample = self.weighted_pick(p)

                if sample > (self.limit-1):
                    sample = 0

                pred = chars[sample]

                if self.sentence_end_token == pred or self.sentence_start_token == pred:
                    ret+='\n'
                    check_action = True
                elif pred == "{" or pred == "}":
                    ret+=pred
                    ret+='\n'
                    if pred =="{":
                        brace_count+=1
                        cnt=2
                    else:
                        brace_count-=1
                elif pred == ";":
                    ret += pred
                    ret += '\n'
                else:
                    ret += pred
                    ret +=" "

                #if check_action != True:
                temp_state.append(sample)
                #else:
                #    check_action = False
                if cnt > 1 and brace_count == 0:
                    checkBrace =True

                char = pred
            else:
                temp_state.append(1)

        state_list.append(np.array(temp_state))

        return ret, state_list[0]

    def get_step(self, chars, vocab, action ,num,limit,sampling_type=1):

        next_state_list=[]
        self.limit = limit
        ret = ""
        temp_ret = []
        char = self.sentence_start_token
        temp_state = []
        check_action = False
        cnt =0
        brace_count = 0
        checkBrace = False
        for n in range(num):
            if checkBrace == False:
                x = np.zeros((1, 1))
                x[0, 0] = vocab[char]
                feed = {self.input_data: x}
                [probs, state] = self.sess.run([self.probs, self.final_state], feed )
                p = probs[0]

                if cnt==0:
                    sample = action
                    cnt+=1
                else:
                    sample = self.weighted_pick(p)


                if sample > (self.limit-1):
                    sample = 0

                pred = chars[sample]

                if self.sentence_end_token == pred or self.sentence_start_token == pred:
                    ret+='\n'
                elif pred == "{" or pred == "}":
                    ret+=pred
                    ret+='\n'
                    if pred =="{":
                        brace_count+=1
                        cnt=2
                    else:
                        brace_count-=1
                elif pred == ";":
                    ret += pred
                    ret += '\n'
                else:
                    ret += pred
                    ret +=" "
                    temp_ret.append(pred)

                temp_state.append(sample)

                if cnt > 1 and brace_count == 0:
                    checkBrace = True
                char = pred
            else:
                temp_state.append(1)

        next_state_list.append(np.array(temp_state))
        return ret, next_state_list[0], self.get_reward(temp_ret)


    def single_get_step(self, chars, vocab, action, num, limit, sampling_type=1):

        next_state_list = []
        self.limit = limit
        ret = ""
        temp_ret = []
        char = self.sentence_start_token
        temp_state = []
        check_action = False
        cnt = 0
        brace_count = 0
        checkBrace = False

        for n in range(num):
            if checkBrace == False:
                x = np.zeros((1, 1))
                x[0, 0] = vocab[char]
                feed = {self.input_data: x}
                [probs, state] = self.sess.run([self.probs, self.final_state], feed)
                p = probs[0]

                if cnt == 0:
                    sample = action
                    cnt += 1
                else:
                    sample = self.weighted_pick(p)

                if sample > (self.limit - 1):
                    sample = 0

                pred = chars[sample]

                if self.sentence_end_token == pred or self.sentence_start_token == pred:
                    ret += '\n'
                elif pred == "{" or pred == "}":
                    ret += pred
                    ret += '\n'
                    if pred == "{":
                        brace_count += 1
                        cnt = 2
                    else:
                        brace_count -= 1
                elif pred == ";":
                    ret += pred
                    ret += '\n'
                else:
                    ret += pred
                    ret += " "
                    temp_ret.append(pred)


                temp_state.append(sample)

                if cnt > 1 and brace_count == 0:
                    checkBrace = True
                char = pred
            else:
                temp_state.append(1)

        next_state_list.append(np.array(temp_state))
        return ret, next_state_list[0], self.get_reward(temp_ret)

    def mThread_cacul_reward(self,sentence):
        return 0


    def get_reward(self, sentence):
        total_cnt = 0.0
        for chars in sentence:
            if chars in self.reward_dict.keys():
                total_cnt+=self.reward_dict[chars]
        return round(total_cnt,1)

    def weighted_pick(self,weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        vocab_number = int(np.searchsorted(t, np.random.rand(1) * s))
        return vocab_number


