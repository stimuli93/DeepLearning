import layers
import numpy as np
import nltk
import initializations


class Word2Vec:
    def __init__(self, size=100, window=5):
        """
        :param size: the required dimension of words
        :param window: the length of context window
        """
        self.size = size
        self.window = window
        self.index_to_word = {}
        self.word_to_index = {}
        self.W_inp = initializations.uniform_init(shape=(3, size))
        self.W_out = initializations.uniform_init(shape=(3, size))

        self.start_token = "START_TOKEN"
        self.end_token = "END_TOKEN"
        self.unknown_token = "UNKNOWN_TOKEN"

        self.index_to_word[0] = self.start_token
        self.index_to_word[1] = self.end_token
        self.index_to_word[2] = self.unknown_token

        self.word_to_index[self.start_token] = 0
        self.word_to_index[self.end_token] = 1
        self.word_to_index[self.unknown_token] = 2

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
        for key in word_count_dict:
            self.index_to_word[itr] = key
            self.word_to_index[key] = itr
            itr += 1

        vocab_size = len(self.index_to_word)
        self.W_inp = initializations.uniform_init(shape=(vocab_size, self.size))
        self.W_out = initializations.uniform_init(shape=(vocab_size, self.size))

    def train(self, X):
        """
        Training based on CBOW model using negative sampling
        :param X: list of sentences used for training
        """
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
                self.train_word_in_context(word, context_window)

    def train_word_in_context(self, word, context_window):
        """
        :param word: the input word
        :param context_window: list of words in the context of given word
        """
        print (word, context_window)