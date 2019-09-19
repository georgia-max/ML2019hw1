"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include an 'alpha' value
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    alpha = params['alpha']                     #1.0

    # Get the train_labels related info
    labels = np.unique(train_labels, return_counts=True)
    counts_each_label = np.asarray(labels[1])   # (20,)
                                                # 480 581 572 587 575
                                                # 592 582 592 596 594
                                                # 598 594 591 594 593
                                                # 599 545 564 464 376
    labels = np.asarray(labels[0])              # (0-19)
    num_of_classes = int(len(labels))                # 20 classes (0-19)

    d, n = train_data.shape     # d=5000 n=11269


    "------------ prior probability P(y)----------------"
    prior_smoothed = np.zeros(num_of_classes)
    prior_smoothed = np.divide((counts_each_label + alpha) , (n + 20 * alpha))
    prior_smoothed_loged = np.log(prior_smoothed)
    np.set_printoptions(precision=3)


    "------------ likelihood  probability P(x|y) ----------------"
    # Loop through every label
    # Create a mask for one label that is (d,n)
    # Mask the label to the train_data
    # Sum all the columns(n) then we get the count of each attribute given one label

    cond_count = np.zeros((1,d), dtype=int)
    for i in range(num_of_classes):  # loop through all labels

        print("Looping thru label: %d" %i)
        # Create masks
        label_mask_row = np.zeros(n)
        for j in range(n):  # loop thru all data
            if train_labels[j] == i:
                label_mask_row[j] = 1
        label_mask = np.tile(label_mask_row, (d,1))
        # Mask the data
        data_masked = np.multiply(train_data, label_mask)
        # Sum the column
        column_sum = np.sum(data_masked, axis=1, dtype=int)
        column_sum = np.reshape(column_sum, (1,d))
        if i == 0:
            cond_count = column_sum
        else:
            cond_count = np.concatenate((cond_count, column_sum), axis=0)

    cond_count = np.transpose(cond_count)   #(5000, 20)

    # Calculate the conditional probability P(x|y)
    # by dividing (#element contains attributes Xi given Yj) by
    #             (#element are labeled yj)
    # cond_count:(5000,20) counts_each_label:(20,)
    cond_prob_smoothed = np.divide((cond_count + alpha), (counts_each_label + 2*alpha))  #(5000,20)


    "------------ Create the return model ----------"
    # :rtype dictionary
    model ={}
    model['prior_prob'] = prior_smoothed
    model['cond_prob'] = cond_prob_smoothed
    model['labels'] = labels

    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    prior_prob = model['prior_prob']
    cond_prob = model['cond_prob']
    labels = model['labels']
    d, n = data.shape
    num_of_labels = labels.size

    data_bar = 1 - data
    cond_prob_bar = 1 - cond_prob
    classification_results = np.zeros(n)


    "------- One instance ---------"
    #Prepare masks
    data_slice = np.reshape(data[:,0], (d,1))
    data_mask = np.tile(data_slice, (1,num_of_labels))
    data_bar_slice = np.reshape(data_bar[:, 0], (d, 1))
    data_bar_mask = np.tile(data_bar_slice, (1,num_of_labels))

    #Apply masks
    cond_prob_masked = np.multiply(data_mask, cond_prob)
    cond_prob_bar_masked = np.multiply(data_bar_mask, cond_prob_bar)
    add_cond_prob = cond_prob_masked + cond_prob_bar_masked

    cond_prob_loged = np.log(add_cond_prob)
    cond_prob_sum = np.sum(cond_prob_loged, axis=0)

    prior_loged = np.log(prior_prob)
    posteriors = np.add(prior_loged, cond_prob_sum)
    classification_results[0] = np.argmax(posteriors)

    for i in range(n):
        # Prepare masks
        data_slice = np.reshape(data[:, i], (d, 1))
        data_mask = np.tile(data_slice, (1, num_of_labels))
        data_bar_slice = np.reshape(data_bar[:, i], (d, 1))
        data_bar_mask = np.tile(data_bar_slice, (1, num_of_labels))

        # Apply masks
        cond_prob_masked = np.multiply(data_mask, cond_prob)
        cond_prob_bar_masked = np.multiply(data_bar_mask, cond_prob_bar)
        add_cond_prob = cond_prob_masked + cond_prob_bar_masked

        cond_prob_loged = np.log(add_cond_prob)
        cond_prob_sum = np.sum(cond_prob_loged, axis=0)

        prior_loged = np.log(prior_prob)
        posteriors = np.add(prior_loged, cond_prob_sum)
        classification_results[i] = np.argmax(posteriors)

    return classification_results
