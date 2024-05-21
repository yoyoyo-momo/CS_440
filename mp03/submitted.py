'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(test, train):
    word_tag_cnt = defaultdict(Counter)
    word_max_tag = {}
    total_tag_cnt = Counter()
    max_tag_cnt = 0
    for sentence in train:
        for word, tag in sentence:
            word_tag_cnt[word][tag] += 1
            if word not in word_max_tag or word_tag_cnt[word][tag] > word_max_tag[word][1]:
                word_max_tag[word] = (tag, word_tag_cnt[word][tag])
            total_tag_cnt[tag] += 1
            if total_tag_cnt[tag] > max_tag_cnt:
                max_tag_cnt = total_tag_cnt[tag]
                max_tag = tag
    predictions = []
    for i in range(len(test)):
        predict = []
        for word in test[i]:
            if word in word_max_tag:
                predict.append((word, word_max_tag[word][0]))
            else:
                predict.append((word, max_tag))
        predictions.append(predict)
    return predictions


def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    epsilon = 0.00001
    def laplace_smoothing(frequency, smoothness):
        probability = defaultdict(dict)
        for y in frequency:
            total_freq = sum(frequency[y].values())
            total_states = len(frequency[y].keys())
            for x in frequency[y].keys():
                probability[y][x] = log((frequency[y][x] + smoothness) / (total_freq + smoothness * (total_states + 1)))
            probability[y]['OOV'] = log(smoothness / (total_freq + smoothness * (total_states + 1)))
        return probability
    
    
    tags_freq = defaultdict(Counter)
    tag_tag_freq = defaultdict(Counter)
    tag_word_freq = defaultdict(Counter)
    
    for sentence in train:
        tags_freq[0][sentence[0][1]] += 1
        tag_word_freq[sentence[0][1]][sentence[0][0]] += 1
        for i in range(1, len(sentence)):
            word, tag = sentence[i]
            tags_freq[0][tag] += 1
            tag_word_freq[tag][word] += 1
            tag_tag_freq[sentence[i - 1][1]][tag] += 1
    tags_prob = laplace_smoothing(tags_freq, epsilon)
    tag_tag_prob = laplace_smoothing(tag_tag_freq, epsilon)
    tag_word_prob = laplace_smoothing(tag_word_freq, epsilon)
    
    predictions = []
    for sentence in test:
        states = defaultdict(dict)
        states[0]['START'] = (0, None)
        
        for t in range(1, len(sentence)):
            for tag in tags_freq[0].keys():
                max_prob = -math.inf
                for prev_tag in states[t - 1].keys():
                    if prev_tag == 'END':
                        continue
                    P_s = states[t - 1][prev_tag][0]
                    P_a = tag_tag_prob[prev_tag]['OOV'] if tag not in tag_tag_prob[prev_tag] else tag_tag_prob[prev_tag][tag]
                    P_b = tag_word_prob[tag]['OOV'] if sentence[t] not in tag_word_prob[tag] else tag_word_prob[tag][sentence[t]]
                    if P_s + P_a + P_b > max_prob:
                        max_prob = P_s + P_a + P_b
                        max_prev_tag = prev_tag
                states[t][tag] = (max_prob, max_prev_tag)
        n = len(sentence)
        predict = []
        max_prob = -math.inf
        for tag in states[n - 1].keys():
            if states[n - 1][tag][0] > max_prob:
                max_prob = states[n - 1][tag][0]
                max_tag = tag
        predict.insert(0, (sentence[n - 1], max_tag))
        prev_state = states[n - 1][max_tag][1]
        for t in range(n - 2, -1, -1):
            cur_state = prev_state
            prev_state = states[t][cur_state][1]
            predict.insert(0, (sentence[t], cur_state))
        predictions.append(predict)
    return predictions

def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    def laplace_smoothing(frequency, smoothness):
        probability = defaultdict(dict)
        for y in frequency:
            total_freq = sum(frequency[y].values())
            total_states = len(frequency[y].keys())
            for x in frequency[y].keys():
                probability[y][x] = log((frequency[y][x] + smoothness) / (total_freq + smoothness * (total_states + 1)))
            probability[y]['OOV'] = log(smoothness / (total_freq + smoothness * (total_states + 1)))
        return probability
    
    def laplace_smoothing_emission(frequency, smoothness, hapax_tag_prob):
        probability = defaultdict(dict)
        for y in frequency:
            hapax_smoothness = smoothness * hapax_tag_prob[y]
            total_freq = sum(frequency[y].values())
            total_states = len(frequency[y].keys())
            for x in frequency[y].keys():
                probability[y][x] = log((frequency[y][x] + hapax_smoothness) / (total_freq + hapax_smoothness * (total_states + 1)))
            probability[y]['OOV'] = log(hapax_smoothness / (total_freq + hapax_smoothness * (total_states + 1)))
        return probability
    
    tags_freq = defaultdict(Counter)
    tag_tag_freq = defaultdict(Counter)
    tag_word_freq = defaultdict(Counter)
    total_word = 0
    
    for sentence in train:
        tags_freq[0][sentence[0][1]] += 1
        tag_word_freq[sentence[0][1]][sentence[0][0]] += 1
        total_word += 1
        for i in range(1, len(sentence)):
            total_word += 1
            word, tag = sentence[i]
            tags_freq[0][tag] += 1
            tag_word_freq[tag][word] += 1
            tag_tag_freq[sentence[i - 1][1]][tag] += 1

    hapax_tag = Counter()
    total_hapax = 0
    for tag in tag_word_freq.keys():
        for word in tag_word_freq[tag].keys():
            if tag_word_freq[tag][word] == 1:
                hapax_tag[tag] += 1
                total_hapax += 1
            
    epsilon = 0.00001
    hapax_tag_prob = {}
    for tag in tags_freq[0].keys():
        hapax_tag_prob[tag] = (hapax_tag[tag] + epsilon) / (total_hapax + epsilon * (len(tags_freq[0].keys()) + 1))
        
    tags_prob = laplace_smoothing(tags_freq, epsilon)
    tag_tag_prob = laplace_smoothing(tag_tag_freq, epsilon)
    tag_word_prob = laplace_smoothing_emission(tag_word_freq, epsilon, hapax_tag_prob)
    
    predictions = []
    for sentence in test:
        states = defaultdict(dict)
        states[0]['START'] = (0, None)
        
        for t in range(1, len(sentence)):
            for tag in tags_freq[0].keys():
                max_prob = -math.inf
                for prev_tag in states[t - 1].keys():
                    if prev_tag == 'END':
                        continue
                    P_s = states[t - 1][prev_tag][0]
                    P_a = tag_tag_prob[prev_tag]['OOV'] if tag not in tag_tag_prob[prev_tag] else tag_tag_prob[prev_tag][tag]
                    P_b = tag_word_prob[tag]['OOV'] if sentence[t] not in tag_word_prob[tag] else tag_word_prob[tag][sentence[t]]
                    if P_s + P_a + P_b > max_prob:
                        max_prob = P_s + P_a + P_b
                        max_prev_tag = prev_tag
                states[t][tag] = (max_prob, max_prev_tag)
        n = len(sentence)
        predict = []
        max_prob = -math.inf
        for tag in states[n - 1].keys():
            if states[n - 1][tag][0] > max_prob:
                max_prob = states[n - 1][tag][0]
                max_tag = tag
        predict.insert(0, (sentence[n - 1], max_tag))
        prev_state = states[n - 1][max_tag][1]
        for t in range(n - 2, -1, -1):
            cur_state = prev_state
            prev_state = states[t][cur_state][1]
            predict.insert(0, (sentence[t], cur_state))
        predictions.append(predict)
    print(predictions[0])
    return predictions