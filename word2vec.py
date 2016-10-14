import layers
import numpy as np
import nltk
import initializations
import pickle
import operator
from collections import Counter
from nltk.corpus import stopwords
from sklearn.utils import shuffle


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
    sim = np.inner(normalise_vector(v1), normalise_vector(v2))
    return sim


class Word2Vec:
    def __init__(self, size=100, window=2, min_count=1):
        """
        :param size: the required dimension of word vectors
        :param window: the length of context window
        :param min_count: minimum frequency required for a word to be considered part of vocabulary
        """
        self.size = size
        self.window = window
        self.index_to_word = {}
        self.word_to_index = {}
        vocab_size = 3
        self.vocab_size = vocab_size
        self.W_inp = initializations.uniform_init(shape=(vocab_size, size))
        self.W_out = initializations.uniform_init(shape=(vocab_size, size))
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
        self.W_inp = initializations.uniform_init(shape=(self.vocab_size, self.size))
        self.W_out = initializations.uniform_init(shape=(self.vocab_size, self.size))

    def train(self, X, learning_rate=1e-2, batch_size=100, nb_epochs=1):
        """
        Training based on CBOW model using negative sampling
        :param nb_epochs: number of iterations
        :param batch_size: the number of sentences trained upon in 1 iteration
        :param learning_rate:
        :param X: list of sentences used for training
        """
        N = len(X)
        start_index = self.word_to_index[self.start_token]
        end_index = self.word_to_index[self.end_token]
        unknown_index = self.word_to_index[self.unknown_token]
        id_x = []
        for i in xrange(N):
            sentence = nltk.word_tokenize(X[i])
            if len(sentence) == 0:
                continue
            id_x.append(start_index)
            for word in sentence:
                if word in self.word_to_index:
                    id_x.append(self.word_to_index[word])
                else:
                    id_x.append(unknown_index)
            id_x.append(end_index)

        corpus_size = len(id_x)
        print corpus_size

        n_iters = corpus_size//batch_size

        for epoch in xrange(nb_epochs):
            for itr in xrange(n_iters):
                batch = np.random.randint(corpus_size, size=batch_size)
                trX = np.zeros([batch_size, self.size])
                trY = np.zeros([batch_size], dtype=np.int32)
                context = []
                ids_to_update = np.zeros([batch_size], dtype=np.int32)
                for id, w_id in enumerate(batch):
                    context_ids = id_x[max(0, w_id-self.window):w_id] + id_x[w_id+1:min(w_id+1+self.window, corpus_size)]
                    context.append(context_ids)
                    context_window = np.array(context_ids)
                    trX[id] = np.mean(self.W_inp[context_window, :], axis=0)
                    trY[id] = id
                    ids_to_update[id] = id_x[w_id]

                context = np.array(context)
                trX, trY, ids_to_update, context = shuffle(trX, trY, ids_to_update, context, random_state=0)
                W = self.W_out[ids_to_update]
                # print trX, trY, ids_to_update, context, W
                b = np.zeros([batch_size])
                layer1, l1cache = layers.dense_forward(trX, W.T, b)
                layer2, l2cache = layers.sigmoid_forward(layer1)
                loss, l3cache = layers.softmax_loss_forward(layer2, trY)
                self.loss_history.append(loss)

                dlayer3 = 1.0
                dlayer2 = layers.softmax_loss_backward(dlayer3, l3cache)
                dlayer1 = layers.sigmoid_backward(dlayer2, l2cache)
                dx_inp, dW_tmp, db = layers.dense_backward(dlayer1, l1cache)
                dW = dW_tmp.T

                for i in xrange(batch_size):
                    self.W_inp[context[i], :] -= (learning_rate * dx_inp[i])/len(context[i])
                self.W_out[ids_to_update, :] -= learning_rate * dW

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
