import codecs
import os
import collections
import csv
import itertools
import nltk
from six.moves import cPickle
import numpy as np
unknown_token = "UNKNOWN_CODE"
sentence_start_token = "CODE_START"
sentence_end_token = "CODE_END"
class CodeLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding # UTF-8

        input_file = os.path.join(data_dir, "allJavaSourceCodeN2.txt")## os.path.join(dir, string) = dir+string
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, 'r',encoding=self.encoding) as f:
            reader = f.readlines()

            # Split full comments into sentences
            sentences = []
            for x in reader:
                sentences.append(sentence_start_token + " " + x + " " + sentence_end_token)

        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        # Limit the code vocabulary sie
        vocabulary_size = 360000
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))
        vocab = word_freq.most_common(vocabulary_size - 1)

        # Get the most common words and build index_to_word and word_to_index vectors
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        self.chars = index_to_word
        self.vocab_size = len(self.chars)
        self.vocab = word_to_index

        x_data =[]
        for sent in tokenized_sentences:
            for w in sent:
                x_data.append(word_to_index[w])

        data = np.array(x_data)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(data)
        np.save(tensor_file, self.tensor)


    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
