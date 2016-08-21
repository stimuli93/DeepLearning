import layers
import numpy as np
import nltk
import initializations
import pickle
import operator
from collections import Counter
from nltk.corpus import stopwords


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
    def __init__(self, size=100, window=2, min_count=1):
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
        stop_words = set(stopwords.words('english'))
        for key, value in word_counter.items():
            if key not in stop_words and value >= self.min_count:
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
        id_x = []
        for i in xrange(N):
            id_x.append(self.word_to_index[self.start_token])
            sentence = nltk.word_tokenize(X[i])
            for word in sentence:
                if word in self.word_to_index:
                    id_x.append(self.word_to_index[word])
            id_x.append(self.word_to_index[self.start_token])

        corpus_size = len(id_x)
        left_window = self.window
        right_window = self.window
        for itr in xrange(n_iters):
            batch = np.random.randint(corpus_size, size=batch_size)
            for w_id in batch:
                loss = 0.0
                context_window = id_x[max(0, w_id-left_window):w_id] +\
                                 id_x[w_id+1:min(w_id+1+right_window, corpus_size)]

                # Negative sampling
                neg_samples_count = 5
                neg_samples = np.random.randint(corpus_size, size=neg_samples_count)
                neg_samples[0] = w_id

                neg_samples = [id_x[sample] for sample in neg_samples]

                loss += self.train_word_in_context(id_x[w_id], np.array(context_window),
                                                   np.array(neg_samples), learning_rate)
                self.loss_history.append(loss)

    def train_word_in_context(self, word_id, context_window, neg_samples, learning_rate):
        """
        :param learning_rate:
        :param word_id: the index of word to predicted
        :param context_window: list of word_ids in the context of given word
        :param neg_samples: list of word_ids choosen as negative samples
        """

        # mean of self.W_inp of words in context_window is used as input
        x_inp = np.mean(self.W_inp[context_window, :], axis=0)
        x_inp = x_inp.reshape((1, self.size))

        neg_sample_count = len(neg_samples)
        b = np.zeros((1, neg_sample_count))
        y = [0]
        output = np.random.randint(neg_sample_count, size=1)
        y[0] = output
        neg_samples[0] = neg_samples[output]
        neg_samples[output] = word_id
        y = np.array(y)
        W = self.W_out[neg_samples, :]

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

    def most_similar(self, word):
        """
        :param word: input word like king
        :return: list of words with similarity which are most similar to the input word
        """
        input_word_vector = self.get_word_vector(word)
        score_list = []
        for key, idx in self.word_to_index.iteritems():
            if word == key:
                continue
            model_word_vector = (self.W_inp[idx] + self.W_out[idx])/2.0
            score_list.append((key, get_vector_similarity(input_word_vector, model_word_vector)))
        score_list.sort(key=operator.itemgetter(1), reverse=True)
        return score_list[:5]

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
