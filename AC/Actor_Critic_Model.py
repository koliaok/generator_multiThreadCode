
import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq



# Replay memory consists of multiple lists of state, action, next state, reward, return from state
replay_states = []
replay_actions = []
replay_rewards = []
replay_next_states = []
replay_return_from_states = []


class Actor:
    def __init__(self ,agent):
        self.hidden_size = 500
        self.save_dir = "Actor_SAVE"
        self.ModelSaveCnt=0
        self.checkpoint_path = os.path.join(self.save_dir, 'Actor_model.ckpt'  )
        self.save_count = 10


        self.agent = agent
        self.observation_space = self.agent.observation_space()
        self.action_space_n = self.agent.action_space()
        # Learning parameters
        self.learning_rate = 0.01
        self.num_layers = 2
        self.batch_size = 1
        self.graph = tf.Graph()


        # Build the graph when instantiated

        with self.graph.as_default():
            def lstm_cell():
                cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
                return cell

            self.cell = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

            with tf.variable_scope('rnn_weight'):
                softmax_w = tf.get_variable("softmax_w1",
                                            [self.observation_space, self.action_space_n],
                                            initializer=tf.contrib.layers.xavier_initializer())
                softmax_b = tf.get_variable("softmax_b1", [self.action_space_n],
                                            initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope('Actor_set_softmax'):
                self.initial_state = self.cell.zero_state(1, tf.float32)

                self.x = tf.placeholder(tf.int32, [self.batch_size, self.observation_space])  # State input
                self.y = tf.placeholder(tf.float32)  # Advantage input

                self.action_input = tf.placeholder("float", [None,self.action_space_n])  # Input action to return the probability associated with that action

                embedding = tf.Variable(tf.random_uniform([self.action_space_n, self.observation_space], -1.0, 1.0))
                self.inputs = tf.nn.embedding_lookup(embedding, self.x)

                outputs, last_state = tf.nn.dynamic_rnn(self.cell, self.inputs,dtype=tf.float32,initial_state=self.initial_state)
                self.outputs = tf.reshape(outputs, [-1, self.observation_space])


                self.policy = self.softmax_policy(self.outputs, softmax_w, softmax_b)  # Softmax policy_ output : action input size


            with tf.variable_scope('Actor_Loss'):
                self.log_action_probability = tf.reduce_sum(self.action_input * tf.log(self.policy))
                self.loss = -self.log_action_probability * self.y  # Loss is score function times advantage
            with tf.variable_scope('Actor_optima'):
                self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            print("Actor Policy Constructed")
            self.init = tf.global_variables_initializer()
            self.ckpt = tf.train.get_checkpoint_state(self.save_dir)
            self.saver = tf.train.Saver(tf.global_variables())


    def softmax_policy(self, state, weights, biases):
        policy = tf.matmul(state, weights) + biases
        policy=tf.reduce_sum(policy,axis=0)
        policy=tf.reshape(policy,[self.batch_size,self.action_space_n])
        policy=tf.divide(policy,tf.reduce_sum(policy))
        policy= tf.nn.softmax(policy)
        return policy


    def actor_start_session(self):
        self.sess = tf.Session(graph=self.graph)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess,self.ckpt.model_checkpoint_path)
            print("Actor Model restor complete")
        else:
            self.sess.run(self.init)



    def rollout_policy(self, timeSteps, episodeNumber):
        """Rollout policy for one episode, update the replay memory and return total reward"""
        total_reward = 0
        curr_state = self.agent.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        episode_return_from_states = []

        for time in range(timeSteps):
            action = self.choose_action(curr_state, time)

            next_state, reward, done = self.agent.single_word_step(action)
            # Update the total reward
            total_reward += reward
            if done or time >= timeSteps:
                break
            # Updating the memory
            self.agent.chars
            print("rollout_policy Implement step : "+ str(time))
            print("reward is : {}".format(reward))
            print("action is : {}".format(action))
            curr_state_l = curr_state.tolist()
            next_state_l = next_state.tolist()
            if curr_state_l not in episode_states:
                episode_states.append(curr_state_l)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_next_states.append(next_state_l)
                episode_return_from_states.append(reward)
                for i in range(len(episode_return_from_states) - 1):
                    episode_return_from_states[i] += reward
            else:
                # Iterate through the replay memory  and update the final return for all states
                for i in range(len(episode_return_from_states)):
                    episode_return_from_states[i] += reward
            curr_state = next_state

        self.update_memory(episode_states, episode_actions, episode_rewards, episode_next_states,
                           episode_return_from_states)
        return episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states, total_reward

    def update_policy(self, advantage_vectors):
        # Update the weights by running gradient descent on graph with loss function defined

        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states

        self.startTime = time.time()
        for i in range(len(replay_states)):

            states = replay_states[i]
            actions = replay_actions[i]
            advantage_vector = advantage_vectors[i]
            for j in range(len(states)):
                action = self.to_action_input(actions[j])

                state = np.asarray(states[j])
                state = state.reshape(1, self.observation_space)

                _, error_value = self.sess.run([self.optim, self.loss],
                                               feed_dict={self.x: state, self.action_input: action,
                                                          self.y: advantage_vector[j]})
            print("Actor Policy Loss : {:.3f}".format(round(float(error_value[0]),5)))
        self.end = time.time()
        print("Actor Policy Update time is : {:.3f}".format(self.end - self.startTime))
        self.ModelSaveCnt+=1
        get = self.ModelSaveCnt%self.save_count
        print("update count : {} ".format(self.ModelSaveCnt))

        if get==0:
            self.actor_save_session()


    def actor_save_session(self):
        self.saver.save(self.sess, self.checkpoint_path, global_step=self.ModelSaveCnt)
        print("save compleate {}".format(self.checkpoint_path))
        print("step is {}".format(self.ModelSaveCnt))


    def choose_action(self, state, time):
        if time % 60 == 0:
            action = 43
        elif time % 45 == 0:
            action = 70
        elif time % 51 == 0:
            action = 237
        elif time % 57== 0:
            action = 37
        elif time % 61== 0:
            action = 25
        else:
            state = state.reshape(1, self.observation_space)
            softmax_out = self.sess.run(self.policy, feed_dict={self.x: state})
            p = np.array(softmax_out[0])
            print(np.sum(p))
            p  = np.divide(p,np.sum(p))
            action = np.random.choice(np.arange(self.action_space_n), 1, replace=True, p=p)[0]  # Sample action from prob density

        return action

    def update_memory(self, episode_states, episode_actions, episode_rewards, episode_next_states,
                      episode_return_from_states):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        # Using first visit Monte Carlo so total return from a state is calculated from first time it is visited

        replay_states.append(episode_states)
        replay_actions.append(episode_actions)
        replay_rewards.append(episode_rewards)
        replay_next_states.append(episode_next_states)
        replay_return_from_states.append(episode_return_from_states)

    def reset_memory(self):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        del replay_states[:], replay_actions[:], replay_rewards[:], replay_next_states[:], replay_return_from_states[:]

    def to_action_input(self, action):
        action_input = [0] * self.action_space_n
        action_input[action] = 1
        action_input = np.asarray(action_input)
        action_input = action_input.reshape(1, self.action_space_n)
        return action_input


class Critic:
    def __init__(self,agent, training=True):


        self.save_dir = "Critic_SAVE"
        self.hidden_size = 500
        self.ModelSaveCnt=0
        self.checkpoint_path = os.path.join(self.save_dir, 'model.ckpt')
        self.ckpt = tf.train.get_checkpoint_state(self.save_dir)
        self.save_count = 10

        self.agent = agent
        self.observation_space =  self.agent.observation_space()
        self.action_space_n = self.agent.action_space()
        self.n_input = self.observation_space
        # Learning Parameters
        self.learning_rate = 0.008
        self.num_epochs = 20
        # Discount factor
        self.discount = 0.90

        self.batch_size = 1
        self.num_layers = 2
        self.graph = tf.Graph()
        # Build the graph when instantiated
        with self.graph.as_default():
            def lstm_cell():
                cell = rnn.BasicLSTMCell(self.hidden_size , state_is_tuple=True)
                return cell

            self.cell = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

            with tf.variable_scope("BasicSoftmax"):
                self.weights = {
                    'h1': tf.Variable(tf.random_normal([self.observation_space, self.action_space_n])),
                    'out': tf.Variable(tf.random_normal([self.action_space_n, 1]))
                }
                self.biases = {
                    'b1': tf.Variable(tf.random_normal([self.action_space_n])),
                    'out': tf.Variable(tf.random_normal([1]))
                }
            with tf.variable_scope("critic_set_out_softmax"):
                self.initial_state = self.cell.zero_state(1, tf.float32)

                self.state_input = tf.placeholder(tf.int32, [self.batch_size, self.observation_space])

                embedding = tf.Variable(tf.random_uniform([self.action_space_n, self.observation_space], -1.0, 1.0))
                self.inputs = tf.nn.embedding_lookup(embedding, self.state_input)


                self.return_input = tf.placeholder("float")  # Target return

                outputs, last_state = tf.nn.dynamic_rnn(self.cell, self.inputs, dtype=tf.float32,
                                                        initial_state=self.initial_state)
                self.outputs = tf.reshape(outputs, [-1, self.observation_space])


                self.value_pred = self.multilayer_perceptron(self.outputs, self.weights, self.biases)

            with tf.variable_scope("critic_loss"):
                self.loss = tf.reduce_mean(tf.pow(self.value_pred - self.return_input, 2))

            with tf.variable_scope("critic_optima"):
                self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.ckpt = tf.train.get_checkpoint_state(self.save_dir)
            self.saver = tf.train.Saver(tf.global_variables())
            print("Value Graph Constructed")

    def critic_start_session(self):
        self.sess = tf.Session(graph=self.graph)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess,self.ckpt.model_checkpoint_path)
            print("critic Model restor complete")
        else:
            self.sess.run(self.init)


    def multilayer_perceptron(self, x, weights, biases):

        layer1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        policy=tf.reduce_sum(layer1,axis=0)
        policy=tf.reshape(policy,[self.batch_size,self.action_space_n])
        policy=tf.divide(policy,tf.reduce_sum(policy))
        out_layer = tf.matmul(policy, weights['out']) + biases['out']

        return out_layer

    def update_value_estimate(self):
        global replay_states, replay_actions, replay_rewards, replay_next_states, replay_return_from_states
        # Monte Carlo prediction
        batch_size = self.batch_size
        get_state=np.ma.size(replay_states,0)
        if get_state < batch_size:
            batch_size = get_state

        for epoch in range(self.num_epochs):
            total_batch = get_state / batch_size
            # Loop over all batches
            for i in (range(int(total_batch))):
                batch_state_input, batch_return_input = self.get_next_batch(batch_size, replay_states,
                                                                            replay_return_from_states)
                # Fit training data using batch
                self.sess.run(self.optim,
                              feed_dict={self.state_input: batch_state_input, self.return_input: batch_return_input})

        self.ModelSaveCnt+=1
        get = self.ModelSaveCnt%self.save_count
        if get==0:
            self.critic_save_session()


    def critic_save_session(self):
        self.saver.save(self.sess, self.checkpoint_path, global_step=self.ModelSaveCnt)
        print("save compleate {}".format(self.checkpoint_path))
        print("step is {}".format(self.ModelSaveCnt))



    def get_advantage_vector(self, states, rewards, next_states):
        # Return TD(0) Advantage for particular state and action
        # Get value of current state
        advantage_vector = []
        for i in range(len(states)):
            state = np.asarray(states[i])
            state = state.reshape(1, self.observation_space)
            next_state = np.asarray(next_states[i])
            next_state = next_state.reshape(1, self.observation_space)
            reward = rewards[i]
            state_value = self.sess.run(self.value_pred, feed_dict={self.state_input: state})
            next_state_value = self.sess.run(self.value_pred, feed_dict={self.state_input: next_state})
            # Current implementation uses TD(0) advantage
            advantage = reward + self.discount * next_state_value - state_value
            advantage_vector.append(advantage)

        return advantage_vector

    def get_next_batch(self, batch_size, states_data, returns_data):
        # Return mini-batch of transitions from replay data
        all_states = []
        all_returns = []
        for i in range(len(states_data)):
            episode_states = states_data[i]
            episode_returns = returns_data[i]
            for j in range(len(episode_states)):
                all_states.append(episode_states[j])
                all_returns.append(episode_returns[j])
        all_states = np.asarray(all_states)
        all_returns = np.asarray(all_returns)
        randidx = np.random.randint(all_states.shape[0], size=batch_size)
        batch_states = all_states[randidx, :]
        batch_returns = all_returns[randidx]
        batch_states_n = np.zeros((1,self.observation_space))
        batch_returns_n = np.zeros((1,1))

        for n in range(batch_size):
            batch_states_n=np.add(batch_states_n,batch_states[n])
            batch_returns_n=batch_returns_n+batch_returns[n]
        batch_states_n = np.divide(batch_states_n,batch_size)
        batch_returns_n = np.divide(batch_returns_n,batch_size)



        return batch_states_n, batch_returns_n