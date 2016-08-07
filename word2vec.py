import layers
import numpy as np
import nltk
import initializations


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
        word_count_dict = {}
        for sentence in vocab:
            word_list = nltk.word_tokenize(sentence)
            for word in word_list:
                word_count_dict[word] = word_count_dict.get(word, 0) + 1

        itr = 3
        for key, value in word_count_dict.iteritems():
            if value > self.min_count:
                self.index_to_word[itr] = key
                self.word_to_index[key] = itr
                itr += 1

        self.vocab_size = len(self.index_to_word)
        self.W_inp = initializations.xavier_init(shape=(self.vocab_size, self.size))
        self.W_out = initializations.xavier_init(shape=(self.vocab_size, self.size))

    def train(self, X, learning_rate=1e-2):
        """
        Training based on CBOW model using negative sampling
        :param X: list of sentences used for training
        """
        # TODO: Process X in batches & run n_iters iterations
        for sentence in X:
            sentence = '%s %s %s' % (self.start_token, sentence, self.end_token)
            word_list = nltk.word_tokenize(sentence)
            word_list_len = len(word_list)
            left_window = self.window // 2
            right_window = self.window - left_window
            for idx, word in enumerate(word_list):
                context_window = []
                j = max(idx - left_window, 0)
                while j < idx:
                    context_window.append(word_list[j])
                    j += 1
                j = idx+1
                while j <= min(word_list_len-1, idx + right_window):
                    context_window.append(word_list[j])
                    j += 1
                self.train_word_in_context(word, context_window, learning_rate)

    def train_word_in_context(self, word, context_window, learning_rate):
        """
        :param word: the input word
        :param context_window: list of words in the context of given word
        """
        # idx of self.unknown_token is 2 so if word is not part of vocab index of unknown_token is used
        context_window_idx = [self.word_to_index.get(wd, 2) for wd in context_window]

        # mean of self.W_inp of words in context_window is used as input
        x_inp = np.mean(self.W_inp[context_window_idx, :], axis=0)
        input_word_idx = self.word_to_index.get(word, 2)

        # Negative sampling
        neg_samples_count = 5
        neg_samples = np.random.choice(self.vocab_size, neg_samples_count, replace=False)

        if input_word_idx not in neg_samples:
            neg_samples[0] = input_word_idx

        W = self.W_out[neg_samples, :]
        x_inp = x_inp.reshape((1, self.size))
        b = np.zeros((1, neg_samples_count))
        y = [0]
        y[0] = neg_samples.tolist().index(input_word_idx)
        y = np.array(y)

        layer1, l1cache = layers.dense_forward(x_inp, W.T, b)
        layer2, l2cache = layers.sigmoid_forward(layer1)
        loss, l3cache = layers.softmax_loss_forward(layer2, y)

        self.loss_history.append(loss)
        dlayer3 = 1.0
        dlayer2 = layers.softmax_loss_backward(dlayer3, l3cache)
        dlayer1 = layers.sigmoid_backward(dlayer2, l2cache)
        dx_inp, dW_tmp, db = layers.dense_backward(dlayer1, l1cache)
        dW = dW_tmp.T

        dx_inp = dx_inp.flatten()
        for id in context_window_idx:
            self.W_inp[id] -= learning_rate*dx_inp

        for itr, id in enumerate(neg_samples):
            self.W_out[id] -= learning_rate*dW[itr]

    def get_word_vector(self, word):
        """
        Returning the learnt vector representation for the given word
        """
        input_word_idx = self.word_to_index.get(word, 2)
        return (self.W_inp[input_word_idx] + self.W_out[input_word_idx])/2.0
