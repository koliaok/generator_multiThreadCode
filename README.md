
Machine Learning based Multi-Threaded Code Generation Technique for Verifying Concurrency Errors in Java Programs
=================================

# **LSTM-RNN + Reinforcement Learning**

Multi-layer Recurrent Neural Networks (LSTM, RNN) for word-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's char-rnn
I change word based LSTM-RNN for learning Java based source code
AC folder is Reinforce Learning of Policy based Actor-Critic Model for making "Multi-Thread Code"

I make the java code generation for multi-thread

=================================

# **Version**

> **Environment**
> - Ubuntu 16.04.3
> - AWS SERVER : p2.xlarge
> - Instance ID : i-0a7bb5380e5312a40
> - Tensorflow 1.3
> - Python3.6
> - CUDA 8.0



# **Discription**

> **Check for source code**
> - I uploaded only the source code because of the size of the training data.
If you want to train, you can get the necessary code in the data directory of the source code structure and try training.
The source code uses two machine learning techniques. 
First, we used LSTM-RNN for code learning. We used the Actor-Critic model, which is one of the reinforcement learning mechanisms, to enhance the behavior of certain codes. Since there are many cases that can be selected as an action in code generation, it is difficult to generate accurate code. However, since it provides basic algorithms and code for other learning, we hope to change it to a more advanced version.
