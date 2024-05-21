'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
from collections import Counter
from collections import defaultdict
import copy
import math

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''
    frequency = {}
    for y in train.keys():
        frequency[y] = Counter()
        for i in range(len(train[y])):
            for j in range(len(train[y][i]) - 1):
                bigram = train[y][i][j] + '*-*-*-*' + train[y][i][j + 1]
                frequency[y][bigram] += 1
    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''
    nonstop = copy.deepcopy(frequency)
    for y in frequency:
        for key in frequency[y].keys():
            text = key.split('*')
            text1, text2 = text[0], text[-1]
            if text1 in stopwords and text2 in stopwords:
                del nonstop[y][key]
    return nonstop
                



def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''
    likelihood = defaultdict(dict)
    for y in nonstop:
        total_bigram = sum(nonstop[y].values())
        total_types = len(nonstop[y].keys())
        for x in nonstop[y].keys():
            likelihood[y][x] = (nonstop[y][x] + smoothness) / (total_bigram + smoothness * (total_types + 1))
        likelihood[y]['OOV'] = smoothness / (total_bigram + smoothness * (total_types + 1))
    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
    for text in texts:
        Ppos = 0
        Pneg = 0
        for i in range(len(text) - 1):
            token1, token2 = text[i], text[i + 1]
            if token1 not in stopwords or token2 not in stopwords:
                bigram = token1 + '*-*-*-*' + token2
                if bigram not in likelihood['pos']:
                    Ppos += math.log(likelihood['pos']['OOV'])
                else:
                    Ppos += math.log(likelihood['pos'][bigram])
                
                if bigram not in likelihood['neg']:
                    Pneg += math.log(likelihood['neg']['OOV'])
                else:
                    Pneg += math.log(likelihood['neg'][bigram])
        Ppos += math.log(prior)
        Pneg += math.log(1 - prior)
        if Ppos > Pneg:
            hypotheses.append('pos')
        elif Ppos < Pneg:
            hypotheses.append('neg')
        else:
            hypotheses.append('undecided')
    return hypotheses



def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for m in range(len(priors)):
        for n in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[n])
            hypotheses = naive_bayes(texts, likelihood, priors[m])
            count_correct = 0
            for (y, yhat) in zip(labels, hypotheses):
                if y==yhat:
                    count_correct += 1
            accuracies[m, n] = count_correct / len(labels)
    return accuracies