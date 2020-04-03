import os
import sys
import numpy as np
from scipy.special import logsumexp
from collections import Counter
import random
import pdb
import math
import itertools
from itertools import product
# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes



def renormalize(cnt):
  '''
  renormalize a Counter()
  '''
  tot = 1. * sum(cnt.values())
  for a_i in cnt:
    cnt[a_i] /= tot
  return cnt

#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------


class NBCPT(object):
  '''
  NB Conditional Probability Table (CPT) for a child attribute.  Each child
  has only the class variable as a parent
  '''

  def __init__(self, A_i):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
        - A_i: the index of the child variable
    DO NOT forget to include the Beta(0.1,0.1) prior
    '''
    self.feature_index = A_i
    self.n_c_0 = 0.2
    self.n_c_1 = 0.2
    self.n_A_i_0_c_0 = 0.1
    self.n_A_i_0_c_1 = 0.1 

  def learn(self, A, C):
    '''
    TODO
    populate any instance variables specified in __init__ to learn
    the parameters for this CPT
        - A: a 2-d numpy array where each row is a sample of assignments
        - C: a 1-d n-element numpy where the elements correspond to the
          class labels of the rows in A
    '''
    self.n_c_0 += sum([int(c == 0) for c in C])
    self.n_c_1 += sum([int(c == 1) for c in C])

    self.n_A_i_0_c_0 += len(C[(A[:, self.feature_index] == 0) & (C == 0)])
    self.n_A_i_0_c_1 += len(C[(A[:, self.feature_index] == 0) & (C == 1)])

  def get_cond_prob(self, entry, c):
    '''
    TODO
    return the conditional probability P(X|Pa(X)) for the values
    specified in the example entry and class label c
        - entry: full assignment of variables
            e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
        - c: the class
    '''
    my_entry = entry[self.feature_index]

    cp_0_0 = self.n_A_i_0_c_0 / self.n_c_0
    cp_1_0 = 1 - (self.n_A_i_0_c_0 / self.n_c_0)
    cp_0_1 = self.n_A_i_0_c_1 / self.n_c_1
    cp_1_1 = 1 - (self.n_A_i_0_c_1 / self.n_c_1)

    cps = [[cp_0_0, cp_1_0], [cp_0_1, cp_1_1]]

    return cps[c][my_entry]




class NBClassifier(object):
  '''
  NB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    Suggestions for the attributes in the classifier:
        - P_c: the probabilities for the class variable C
        - cpts: a list of NBCPT objects
    '''
    self.NBCPT_list = [NBCPT(i) for i in range(A_train.shape[1])]
    self.p_c_0 = 0.0
    self._train(A_train, C_train)

  def _train(self, A_train, C_train):
    '''
    TODO
    train your NB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
        - A_train: a 2-d numpy array where each row is a sample of assignments
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A
    '''

    self.p_c_0 = sum([c == 0 for c in C_train]) / len(C_train)
    for ncbpt in self.NBCPT_list:
      ncbpt.learn(A_train, C_train)



  def classify(self, entry):
    '''
    TODO
    return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry
    - entry: full assignment of variables
    e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1
    NOTE this must return both the predicated label {0,1} for the class
    variable and also the log of the conditional probability of this
    assignment in a tuple, e.g. return (c_pred, logP_c_pred)
    ''' 

    if -1 in entry:
      empty_idx = []
      n_empty = 0
      for i,e in enumerate(entry):
        if e == -1 :
          n_empty += 1
          empty_idx.append(i)

      assignments = list(itertools.product((0, 1), repeat=n_empty))
      p_c_0_a_s = 0
      p_c_1_a_s = 0


      for assignment in assignments:
        for i in range(n_empty):
          entry[empty_idx[i]] = assignment[i]
        
        p_c_0_a = self.p_c_0
        p_c_1_a = 1 - self.p_c_0

        for i in range(len(entry)):
          p_c_0_a = p_c_0_a * self.NBCPT_list[i].get_cond_prob(entry, 0)

        for i in range(len(entry)):
          p_c_1_a = p_c_1_a * self.NBCPT_list[i].get_cond_prob(entry, 1)

        p_c_0_a_s += p_c_0_a
        p_c_1_a_s += p_c_1_a

      pred = (max(p_c_0_a_s, p_c_1_a_s) / (p_c_0_a_s + p_c_1_a_s))
      pred_class = np.argmax([p_c_0_a_s, p_c_1_a_s])

    else:
      p_c_0_a = self.p_c_0
      p_c_1_a = 1 - self.p_c_0

      for i in range(len(entry)):
        p_c_0_a = p_c_0_a * self.NBCPT_list[i].get_cond_prob(entry, 0)

      for i in range(len(entry)):
        p_c_1_a = p_c_1_a * self.NBCPT_list[i].get_cond_prob(entry, 1)

      pred = (max(p_c_0_a, p_c_1_a) / (p_c_0_a + p_c_1_a))
      pred_class = np.argmax([p_c_0_a, p_c_1_a])

    return(pred_class, np.log(pred))


  def predict_unobserved(self, entry, index):
    '''
    TODO
    Predicts P(A_index = 1 \mid entry)
    '''

    empty_idx = []
    n_empty = 0
    for i,e in enumerate(entry):
      if i != index:
        if e == -1 :
          n_empty += 1
          empty_idx.append(i)

    assignments = list(itertools.product((0, 1), repeat=n_empty))


    entry_copy = np.copy(entry)
    tmp = [0,0]
    for j in range(2):
      p_c_0_a_s = 0
      p_c_1_a_s = 0
      entry = np.copy(entry_copy)
      entry[index] = j
      for assignment in assignments:
        for i in range(n_empty):
          entry[empty_idx[i]] = assignment[i]
        
        p_c_0_a = self.p_c_0
        p_c_1_a = 1 - self.p_c_0

        for i in range(len(entry)):
          p_c_0_a = p_c_0_a * self.NBCPT_list[i].get_cond_prob(entry, 0)

        for i in range(len(entry)):
          p_c_1_a = p_c_1_a * self.NBCPT_list[i].get_cond_prob(entry, 1)

        p_c_0_a_s += p_c_0_a
        p_c_1_a_s += p_c_1_a

      tmp[j] += (p_c_0_a_s + p_c_1_a_s)

    return tmp/sum(tmp)


    



# load all data
A_base, C_base = load_vote_data()


def evaluate(classifier_cls, train_subset=False):
  '''
  evaluate the classifier specified by classifier_cls using 10-fold cross
  validation
  - classifier_cls: either NBClassifier or other classifiers
  - train_subset: train the classifier on a smaller subset of the training
    data
  NOTE you do *not* need to modify this function
  '''
  global A_base, C_base

  A, C = A_base, C_base

  # score classifier on specified attributes, A, against provided labels,
  # C
  def get_classification_results(classifier, A, C):
    results = []
    pp = []
    for entry, c in zip(A, C):
      c_pred, unused = classifier.classify(entry)
      results.append((c_pred == c))
      pp.append(unused)
    # print('logprobs', np.array(pp))
    return results
  # partition train and test set for 10 rounds
  M, N = A.shape
  tot_correct = 0
  tot_test = 0
  step = int(M / 10 + 1)
  for holdout_round, i in enumerate(range(0, M, step)):
    # print("Holdout round: %s." % (holdout_round + 1))
    A_train = np.vstack([A[0:i, :], A[i+step:, :]])
    C_train = np.hstack([C[0:i], C[i+step:]])
    A_test = A[i: i+step, :]
    C_test = C[i: i+step]
    if train_subset:
      A_train = A_train[: 16, :]
      C_train = C_train[: 16]


    # train the classifiers
    classifier = classifier_cls(A_train, C_train)

    train_results = get_classification_results(classifier, A_train, C_train)
    # print(
    #    '  train correct {}/{}'.format(np.sum(train_results), A_train.shape[0]))
    test_results = get_classification_results(classifier, A_test, C_test)
    tot_correct += sum(test_results)
    tot_test += len(test_results)

  return 1.*tot_correct/tot_test, tot_test


def evaluate_incomplete_entry(classifier_cls):

  global A_base, C_base

  # train a classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  c_pred, logP_c_pred = classifier.classify(entry)
  print("  P(C={}|A_observed) = {:2.4f}".format(c_pred, np.exp(logP_c_pred)))

  return


def predict_unobserved(classifier_cls, index=11):
  global A_base, C_base

  # train a classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  a_pred = classifier.predict_unobserved(entry, index)
  print("  P(A{}=1|A_observed) = {:2.4f}".format(index+1, a_pred[1]))

  return


def main():

  '''
  TODO modify or add calls to evaluate() to evaluate your implemented
  classifiers
  Suggestions on how to use:
  ##For Q1
  print('Naive Bayes')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print('  10-fold cross validation total test error {:2.4f} on {} '
        'examples'.format(1 - accuracy, num_examples))
  ##For Q2
  print('Naive Bayes Classifier on missing data')
  evaluate_incomplete_entry(NBClassifier)

  index = 11
  print('Prediting vote of A%s using NBClassifier on missing data' % (
      index + 1))
  predict_unobserved(NBClassifier, index)
  ##For Q3
  print('Naive Bayes (Small Data)')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
  print('  10-fold cross validation total test error {:2.4f} on {} '
        'examples'.format(1 - accuracy, num_examples))

  '''

  ##For Q1
  print('Naive Bayes')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print('  10-fold cross validation total test error {:2.4f} on {} '
        'examples'.format(1 - accuracy, num_examples))

  ##For Q2
  print('Naive Bayes Classifier on missing data')
  evaluate_incomplete_entry(NBClassifier)


  index = 11
  print('Prediting vote of A%s using NBClassifier on missing data' % (
      index + 1))
  predict_unobserved(NBClassifier, index)


  ##For Q3
  print('Naive Bayes (Small Data)')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
  print('  10-fold cross validation total test error {:2.4f} on {} '
        'examples'.format(1 - accuracy, num_examples))
  

if __name__ == '__main__':

  main()
