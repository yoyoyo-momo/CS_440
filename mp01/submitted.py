'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import collections
import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    N = len(texts)
    cX0 = 0
    dic = {}
    for text in texts:
      cnt = text.count(word0)
      if cnt not in dic:
        dic[cnt] = 1
      else:
        dic[cnt] += 1
      cX0 = max(cX0, cnt)
    Pmarginal = [0] * (cX0 + 1)
    for i in range(cX0 + 1):
      if i not in dic:
        Pmarginal[i] = 0
      else:
        Pmarginal[i] = dic[i] / N
    Pmarginal = np.asarray(Pmarginal)
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    
    cX0 = cX1 = 0
    cnt_0 = []
    cnt_1 = []
    
    for text in texts:
      x0 = text.count(word0)
      cnt_0.append(x0)
      cX0 = max(cX0, x0)
      x1 = text.count(word1)
      cnt_1.append(x1)
      cX1 = max(cX1, x1)
    dis = [[0] * (cX1 + 1) for _ in range(cX0 + 1)]
    for i in range(len(cnt_0)):
      dis[cnt_0[i]][cnt_1[i]] += 1
    
    Pcond = [[0] * (cX1 + 1) for _ in range(cX0 + 1)]
    for x0 in range(cX0 + 1):
      total_0 = sum(dis[x0])
      for x1 in range(cX1 + 1):
        if total_0 == 0:
          Pcond[x0][x1] = np.nan
        else:
          Pcond[x0][x1] = dis[x0][x1] / total_0
    Pcond = np.asarray(Pcond)
        
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    cX0 = Pcond.shape[0]
    cX1 = Pcond.shape[1]
    Pjoint = [[0] * cX1 for _ in range(cX0)]
    for x0 in range(cX0):
      for x1 in range(cX1):
        if Pmarginal[x0] != 0:
            Pjoint[x0][x1] = Pmarginal[x0] * Pcond[x0][x1]
    Pjoint = np.asarray(Pjoint)
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    
    cX0, cX1 = Pjoint.shape[0], Pjoint.shape[1]
    EX0 = EX1 = 0
    for x0 in range(cX0):
      EX0 += x0 * sum(Pjoint[x0])
    for x1 in range(cX1):
      sum_x0 = 0
      for x0 in range(cX0):
        sum_x0 += Pjoint[x0][x1]
      EX1 += sum_x0 * x1
    mu = np.array([EX0, EX1])
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    cX0, cX1 = Pjoint.shape[0], Pjoint.shape[1]
    
    EX00 = EX01 = EX11 = 0
    for x0 in range(cX0):
      EX00 += ((x0 - mu[0]) ** 2) * sum(Pjoint[x0])
    for x1 in range(cX1):
      sum_x0 = 0
      for x0 in range(cX0):
        sum_x0 += Pjoint[x0][x1]
        EX01 += Pjoint[x0][x1] * (x0 - mu[0]) * (x1 - mu[1])
      EX11 +=  ((x1 - mu[1]) ** 2) * sum_x0
    Sigma = np.array([[EX00, EX01], [EX01, EX11]])  
    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    Pfunc = collections.defaultdict(int)
    cX0, cX1 = Pjoint.shape[0], Pjoint.shape[1]
    for x0 in range(cX0):
      for x1 in range(cX1):
        Pfunc[f(x0, x1)] += Pjoint[x0][x1]
        
    return Pfunc
    
