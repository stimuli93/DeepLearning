import layers
import numpy as np
import nltk
import initializations
import pickle
from collections import Counter


def normalise_vector(x):
    """
    :param x: vector to be normalised
    :return: normalised vector
    """
    magnitude = np.sqrt(np.sum(x**2))
    return x/magnitude


def get_vector_similarity(v1, v2):
    """
    :param v1: Input vector1
    :param v2: Input vector2
    :return:  similarity b/w vector v1 & v2
    """
    sim = np.inner(normalise_vector(v1),normalise_vector(v2))
    return sim


class Word2Vec:
    def __init__(self, size=100, window=5, min_count=1):
        """
        :param size: the required dimension of words
        :param window: the length of context window
        """
        self.size = size
        self.window = window
        self.index_to_word = {}
        self.word_to_index = {}
        self.W_inp = initializations.xavier_init(shape=(3, size))
        self.W_out = initializations.xavier_init(shape=(3, size))
        self.min_count = min_count

        self.start_token = "START_TOKEN"
        self.end_token = "END_TOKEN"
        self.unknown_token = "UNKNOWN_TOKEN"

        self.index_to_word[0] = self.start_token
        self.index_to_word[1] = self.end_token
        self.index_to_word[2] = self.unknown_token

        self.word_to_index[self.start_token] = 0
        self.word_to_index[self.end_token] = 1
        self.word_to_index[self.unknown_token] = 2
        self.vocab_size = 3
        self.loss_history = []

    def build_vocab(self, vocab):
        """
        :param vocab: list of sentences to be used for training
        """
        word_counter = Counter()
        for sentence in vocab:
            word_list = nltk.word_tokenize(sentence)
            for word in word_list:
                word_counter[word] += 1

        itr = 3
        for key, value in word_counter.items():
            if value > self.min_count:
                self.index_to_word[itr] = key
                self.word_to_index[key] = itr
                itr += 1

        self.vocab_size = len(self.index_to_word)
        self.W_inp = initializations.xavier_init(shape=(self.vocab_size, self.size))
        self.W_out = initializations.xavier_init(shape=(self.vocab_size, self.size))

    def train(self, X, learning_rate=1e-2, batch_size=20, n_iters=50):
        """
        Training based on CBOW model using negative sampling
        :param n_iters: number of iterations
        :param batch_size: the number of sentences trained upon in 1 iteration
        :param learning_rate:
        :param X: list of sentences used for training
        """
        N = len(X)
        left_window = self.window // 2
        right_window = self.window - left_window
        for itr in xrange(n_iters):
            batch = np.random.randint(N, size=batch_size)
            for b_id in batch:
                sentence = '%s %s %s' % (self.start_token, X[b_id], self.end_token)
                word_list = nltk.word_tokenize(sentence)
                word_idx = [self.word_to_index.get(wd, 2) for wd in word_list]
                word_list_len = len(word_idx)
                loss = 0.0
                for i in xrange(word_list_len):
                    context_window = word_idx[max(0, i-left_window):i] +\
                                     word_idx[i+1:min(i+1+right_window, word_list_len)]
                    loss += self.train_word_in_context(word_idx[i], context_window, learning_rate)
                self.loss_history.append(loss/word_list_len)

    def train_word_in_context(self, word_id, context_window, learning_rate):
        """
        :param learning_rate:
        :param word_id: the index of word to predicted
        :param context_window: list of word_ids in the context of given word
        """
        # mean of self.W_inp of words in context_window is used as input
        x_inp = np.mean(self.W_inp[context_window, :], axis=0)

        # Negative sampling
        neg_samples_count = 5
        neg_samples = np.random.randint(self.vocab_size, size=neg_samples_count)

        if word_id not in neg_samples:
            neg_samples[0] = word_id
        
        W = self.W_out[neg_samples, :]
        x_inp = x_inp.reshape((1, self.size))
        b = np.zeros((1, neg_samples_count))
        y = [0]
        y[0] = neg_samples.tolist().index(word_id)
        y = np.array(y)

        layer1, l1cache = layers.dense_forward(x_inp, W.T, b)
        layer2, l2cache = layers.sigmoid_forward(layer1)
        loss, l3cache = layers.softmax_loss_forward(layer2, y)

        dlayer3 = 1.0
        dlayer2 = layers.softmax_loss_backward(dlayer3, l3cache)
        dlayer1 = layers.sigmoid_backward(dlayer2, l2cache)
        dx_inp, dW_tmp, db = layers.dense_backward(dlayer1, l1cache)
        dW = dW_tmp.T

        dx_inp = dx_inp.flatten()
        self.W_inp[context_window] -= learning_rate*dx_inp
        self.W_out[neg_samples] -= learning_rate*dW
        return loss

    def get_word_vector(self, word):
        """
        Returning the learnt vector representation for the given word
        """
        input_word_idx = self.word_to_index.get(word, 2)
        return (self.W_inp[input_word_idx] + self.W_out[input_word_idx])/2.0

    def save_model(self, filename):
        """
        Store model parameters in the specified filename
        """
        with open(filename, 'wb') as fp:
            pickle.dump(self.index_to_word, fp)
            pickle.dump(self.word_to_index, fp)
            pickle.dump(self.W_inp, fp)
            pickle.dump(self.W_out, fp)

    def load_model(self, filename):
        """
        :param filename:
        :return:
        """
        with open(filename, 'rb') as fp:
            self.index_to_word = pickle.load(fp)
            self.word_to_index = pickle.load(fp)
            self.W_inp = pickle.load(fp)
            self.W_out = pickle.load(fp)
        self.vocab_size = len(self.index_to_word)
