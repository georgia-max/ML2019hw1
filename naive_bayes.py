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
    alpha = params['alpha']

    labels = np.unique(train_labels)

    # trainlabels = 11269
    # labels = 20
    d, n = train_data.shape
    num_classes = labels.size

    "------------ prior probability P(y)----------------"
    prior_smoothed = np.zeros(n)
    label_count =np.zeros(20)


    for i in range(len(train_labels)):
        for j in range (0,20):
            if train_labels[i] == j:
                label_count[j] += 1

    print(len(train_labels), label_count)
    print(sum(label_count))

    prior_smoothed = (label_count + alpha) / (len(train_labels) + 20 * alpha)
    print(prior_smoothed, sum(prior_smoothed))
    # TODO: Make it a dictionary if possible
    "------------ likelihood  probability P(x|y) ----------------"

    find_class = np.zeros(len(train_labels))

    for ii in range(len(train_labels)):
        # 0-11269
        if train_labels[ii] == 0:
            find_class[ii] = 1
        else:
            find_class[ii] = 0
    find_class = find_class.reshape((len(train_labels), 1))
    count_xy = np.dot(train_data, find_class)
    print("countxy\n", count_xy)
    print(num_classes)


    for i in range(1,num_classes):
      # 0-19
        for ii in range(len(train_labels)):
            # 0-11269
            if train_labels[ii]== i:
                find_class[ii] =1
            else:
                find_class[ii] = 0
        find_class= find_class.reshape((len(train_labels),1))
        count_xy = np.append(count_xy,  np.dot(train_data, find_class),axis =1)
    #print(count_xy.shape)
    #print(count_xy)
    #print(type(count_xy))

    "---------------- P(y|x)------------"
    #Reshape label_count
    label_count = np.array([label_count.transpose()] * d)
    print ("reshape\n", label_count, label_count.shape)

    prob_xgiveny_smoothed = np.divide((count_xy + alpha),  (label_count + 2 * alpha))
    print("shape:", prob_xgiveny_smoothed.shape, count_xy.shape, label_count.shape)
    print("prob xy smoothed\n", prob_xgiveny_smoothed)

    testarray = np.sum(prob_xgiveny_smoothed, axis=0)
    print('testarray',testarray)
    print('testarray', testarray.shape)


    model ={}
    model['prior_prob']= prior_smoothed
    model['cond_prob']= prob_xgiveny_smoothed
    #print(model)
    print('prior shape', model['prior_prob'].shape)
    print('con_prob',model['cond_prob'].shape )


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


    data_bar = 1 - data
    cond_prob = model['cond_prob']
    cond_prob_bar = 1 - cond_prob
    print("data ", data)
    print("data_bar", data_bar)
    classification_result = np.zeros(np.size(data,1))
    print(classification_result.shape)

    print("cond prob", cond_prob)
    print("cond prob shape", cond_prob.shape)
    print("cond_prob_bar",cond_prob_bar)
    print("cond_prob_bar shape", cond_prob_bar.shape)


    # One instance
    data_mask = np.array([data[:, 0], ] * 20).transpose()
    #print("data", data, data[:, 0], data[0, :])
    print("data mask\n", data_mask.shape)
    data_bar_mask = np.array([data_bar[:, 0], ] * 20).transpose()
    print("data mask bar\n", data_bar_mask.shape)
    cond_prob_masked = np.multiply(data_mask, cond_prob)

    print("cond_prob_masked", cond_prob_masked)
    cond_prob_bar_masked = np.multiply(data_bar_mask, cond_prob_bar)
    print("cond prob bar masked", cond_prob_bar_masked)
    add_cond_prob = cond_prob_masked + cond_prob_bar_masked
    print("add_cond_prob",add_cond_prob)
    cond_prob_loged = np.log(add_cond_prob)
    print("loged", cond_prob_loged)
    cond_prob_sum = np.sum(cond_prob_loged, axis=0)
    print("cond_prob sum", cond_prob_sum, cond_prob_sum.shape)

    posteriors = np.multiply(model['prior_prob'], cond_prob_sum)
    print("posterior", posteriors)
    classification_results = np.argmax(posteriors)
    print("class", classification_results)


    for i in range(np.size(data,1)):
        data_mask = np.array([data[:, i], ] * 20).transpose()
        data_bar_mask = np.array([data_bar[:, i], ] * 20).transpose()
        cond_prob_masked = np.multiply(data_mask, cond_prob)
        cond_prob_bar_masked =  np.multiply(data_bar_mask, cond_prob_bar)
        cond_prob_loged = np.log(cond_prob_masked + cond_prob_bar_masked)
        cond_prob_sum = np.sum(cond_prob_loged,axis= 0)

        posterior = np.multiply(model['prior_prob'], cond_prob_sum)
        classification_result[i] = np.argmax(posterior)

    print(classification_result[0:20])
    return classification_result




    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA
