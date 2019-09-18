"""This module contains a utility function to load 20news data."""
import numpy as np
from scipy.sparse import dok_matrix, csc_matrix
import pickle
import os
import time


def load_all_data():
    """Load 20news data from raw text or from a cached Python pickle file.

    :return: tuple containing num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels
    :rtype: tuple
    """
    pickle_file = 'loaded_20news.pkl'

    start_time = time.time()

    if os.path.exists(pickle_file):

        print("Found pickle file. Loading 20news data from file.")
        print("Doing so should be faster than loading from raw text, but if the file is corrupted, "
                "delete it and this script will automatically load from the raw text next time it is run.")

        with open(pickle_file, "rb") as in_file:
            dictionary = pickle.load(in_file)

        num_words = dictionary['num_words']
        num_training = dictionary['num_training']
        num_testing = dictionary['num_testing']
        train_data = dictionary['train_data']
        test_data = dictionary['test_data']
        train_labels = dictionary['train_labels']
        test_labels = dictionary['test_labels']

    else:

        print("Pickled file does not exist yet. Loading data from raw text files.")

        train_data_ijv = np.loadtxt('20news-bydate/matlab/train.data', dtype=int)
        test_data_ijv = np.loadtxt('20news-bydate/matlab/test.data', dtype=int)

        num_training = train_data_ijv[:, 0].max()
        num_testing = test_data_ijv[:, 0].max()

        # convert to zero-indexing
        train_data_ijv[:, :2] -= 1
        test_data_ijv[:, :2] -= 1

        assert train_data_ijv.min() >= 0, "Indexing correction created a negative index"
        assert test_data_ijv.min() >= 0, "Indexing correction created a negative index"

        max_word = max(train_data_ijv[:, 1].max(), test_data_ijv[:, 1].max())

        num_words = max_word + 1

        # set up data matrices

        train_data = dok_matrix((num_words, num_training), dtype=bool)
        for (col, row, val) in train_data_ijv:
            train_data[row, col] = (val > 0)

        test_data = dok_matrix((num_words, num_testing), dtype=bool)
        for (col, row, val) in test_data_ijv:
            test_data[row, col] = (val > 0)

        train_data = csc_matrix(train_data)
        test_data = csc_matrix(test_data)

        # load labels and convert to zero-indexed

        train_labels = np.loadtxt('20news-bydate/matlab/train.label') - 1
        test_labels = np.loadtxt('20news-bydate/matlab/test.label') - 1

        # save loaded objects to cache file

        dictionary = dict()

        dictionary['num_words'] = num_words
        dictionary['num_training'] = num_training
        dictionary['num_testing'] = num_testing
        dictionary['train_data'] = train_data
        dictionary['test_data'] = test_data
        dictionary['train_labels'] = train_labels
        dictionary['test_labels'] = test_labels

        with open(pickle_file, "wb") as out_file:
            pickle.dump(dictionary, out_file)

    print("Finished loading in %2.2f seconds." % (time.time() - start_time))

    return num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels
