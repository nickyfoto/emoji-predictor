"""

To plot confusion matrix with emoji
You need to install plotly
https://plot.ly/python/static-image-export/
https://github.com/plotly/orca#installation


"""



print(__doc__)


from time import time
from sklearn import metrics
from sklearn.utils.extmath import density
import numpy as np
import plotly.figure_factory as ff
from itertools import count
c = count()

def benchmark(clf, X_train, y_train, X_test, y_test, mapping, opts=None,
              feature_names=None, target_names=None):
    """
    Args:
        X_train: Vectorized training data
        y_train: label, 1 dimensional
        X_test: Vectorized testing data
        y_test: label, 1 dimensional
        mapping: DataFrame read from Mapping.csv see get_data.py
        opts: op.parse_args(argv), optional
    Returns:
        clf_descr, score, train_time, test_time, t_score
    """
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)


    t_preds = clf.predict(X_train)
    
    t_score = metrics.accuracy_score(y_train, t_preds)
    print("Training accuracy:   %0.3f" % t_score)

    score = metrics.accuracy_score(y_test, pred)
    print("Testing accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts and opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print("%s: %s" % (label, " ".join(feature_names[top10])))
        print()

    if opts and opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if opts and opts.print_cm:
        print("confusion matrix:")
        cm = metrics.confusion_matrix(y_test, pred)
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 
              decimals=2)
        print(cm)
        # print(cm.sum(axis=1))
        fig = ff.create_annotated_heatmap(z=cm,
                                          x=mapping.emoticons.tolist(),
                                          y=mapping.emoticons.tolist(),
                                          showscale=True)
        # note the cm plots are still saved at data_visualization folder
        fig.write_image(f'../data_visualization/cm_plots/model_{next(c)}_{str(clf)[:10]}_cm.png', width=800, height=600)


    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time, t_score