import logging
import numpy as np
import pandas as pd
from optparse import OptionParser
import sys
from time import time
import matplotlib, sklearn, plotly

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from ct import CleanTweet, process_text
from bm import benchmark
from utils import get_data, get_top
import plotly.figure_factory as ff


def get_versions():
    print('Version information')
    print('python: {}'.format(sys.version))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('numpy: {}'.format(np.__version__))
    print('pandas: {}'.format(pd.__version__))
    print('plotly: {}'.format(plotly.__version__))

get_versions()
# Version information
# python: 3.7.4 (default, Sep  7 2019, 18:27:02)
# [Clang 10.0.1 (clang-1001.0.46.4)]
# matplotlib: 3.1.1
# numpy: 1.17.2
# pandas: 0.25.1
# plotly: 4.1.1

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_emojis",
              action="store_true", dest="all_emojis",
              help="Whether to predict all emojis or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")



def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

op.print_help()
print()


# #############################################################################
# Load some categories from the training set
print("Loading emoji dataset:")


df, mapping = get_data()
df = process_text(df)



if opts.all_emojis:
    emojis = get_top(df, mapping, n=len(mapping))
    data = df
else:
    emojis = get_top(df, mapping, n=5)
    data = df[df.label.isin(emojis.index)]

mapping = emojis

target_names = emojis.emoticons.to_list()
print(target_names)

X_train, X_test, y_train, y_test = train_test_split(data.text, data.label, 
                                                    test_size=0.2, 
                                                    random_state=42,
                                                    stratify=data.label)

print('data loaded')


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


data_train_size_mb = size_mb(X_train)
data_test_size_mb = size_mb(X_test)

print("%d documents - %0.3fMB (training set)" % (
    len(X_train), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(X_test), data_test_size_mb))
print("%d emojis" % len(emojis))
print()


print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
# ct = CleanTweet()
# X_train = ct.transform(X_train)
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train_counts = vectorizer.transform(X_train)
else:
    # vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english')
    vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english')
    X_train_counts = vectorizer.fit_transform(X_train)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train_counts.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test_counts = vectorizer.transform(X_test)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test_counts.shape)
print()

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train_counts, y_train)
    X_test = ch2.transform(X_test_counts)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print(feature_names)
    print(ch2.scores_)
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

X_train = X_train_counts
X_test = X_test_counts

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50, tol=1e-3),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto'), 'logit'),
        (RandomForestClassifier(n_estimators=10), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, X_train, y_train, X_test, y_test, mapping, opts, 
                   feature_names=feature_names, target_names=target_names))


for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3), X_train, y_train, X_test, y_test, mapping, opts,
    feature_names=feature_names, target_names=target_names))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                           penalty=penalty), X_train, y_train, X_test, y_test, mapping, opts,
    feature_names=feature_names, target_names=target_names))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet"), X_train, y_train, X_test, y_test, mapping, opts,
feature_names=feature_names, target_names=target_names))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid(), X_train, y_train, X_test, y_test, mapping, opts,feature_names=feature_names, target_names=target_names))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01), X_train, y_train, X_test, y_test, mapping, opts,feature_names=feature_names, target_names=target_names))
results.append(benchmark(BernoulliNB(alpha=.01), X_train, y_train, X_test, y_test, mapping, opts,feature_names=feature_names, target_names=target_names))
results.append(benchmark(ComplementNB(alpha=.1), X_train, y_train, X_test, y_test, mapping, opts,feature_names=feature_names, target_names=target_names))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))]), X_train, y_train, X_test, y_test, mapping, opts,feature_names=feature_names, target_names=target_names))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(5)]

clf_names, score, training_time, test_time, t_score = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .25, t_score, .2, label="t_score", color='m')
plt.barh(indices + .5, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .75, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.savefig(f'{len(emojis)}.png')
