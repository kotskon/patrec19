import config
from src import datautilus
from src import euclidean_classifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
import numpy as np

#A global variable.
dataHouse = config.ToolConfig ()

def main ():
    """
    This function works as a general orchestrator of the exercise.
    """
    print ('Module initiated. Reading data...')

    #Step 1
    X_train, X_test, y_train, y_test = datautilus.dataReader (
                                         dataHouse.train,
                                         dataHouse.test)
    print ('All aboard!')

    #Step 2
    print ('Picking pattern #131...')
    feat_131, id_131 = datautilus.pickDigit (X_train, y_train, 131)
    print ('Done! Number is ', int (id_131))
    datautilus.printSingleDigit (feat_131, dataHouse.save, 'step2')

    print ('Commencing serial digit search...')
    posx, posy = 0, 0
    step3_fig, axs3 = plt.subplots (2, 5, figsize = (20, 20))
    step9_fig, axs9 = plt.subplots (2, 5, figsize = (20, 20))

    #Step 10
    for num in range(10):
        #Step 3
        step3_num = datautilus.findDigit (X_train, y_train, num)[0, :]
        #Steps 6-9
        mean_num, var_num = datautilus.analyzeDigit (X_train, y_train, num,
                                                     dataHouse.save)
        if num == 0:
            means_overall = mean_num
            vars_overall = var_num
        else:
            means_overall = np.vstack ((means_overall, mean_num))
            vars_overall = np.vstack ((vars_overall, var_num))

        axs3[posx, posy], im3 = datautilus.digit2Fig (step3_num,
                                                      axs3[posx, posy])
        axs9[posx, posy], im9 = datautilus.digit2Fig (mean_num,
                                                      axs9[posx, posy])
        step9_fig.colorbar (im9, ax = axs9[posx, posy], fraction = .04)
        posy += 1
        if posy % 5 == 0:
            posy = 0
            posx += 1

    step3_fig.colorbar (im3, ax = axs3, orientation = 'horizontal',
                        fraction = .025)
    step3_fig.savefig(dataHouse.save + 'step3.svg')
    plt.close ()
    step9_fig.tight_layout ()
    step9_fig.savefig(dataHouse.save + 'step9.svg')
    #plt.show ()
    plt.close ()

    #Steps 4-5
    flat_idx = 9 * 16 + 9
    print ('For digit 0, mean value of pixel (10, 10) is ',
           means_overall[0, flat_idx])
    print ('For digit 0, variance of pixel (10, 10) is ',
           vars_overall[0, flat_idx])

    #Step 10
    feat_101, id_101 = datautilus.pickDigit (X_train, y_train, 101)
    pred_101 = datautilus.askEuclid (means_overall, feat_101)
    print ('prediction: ', pred_101)
    print ('actual: ', int (id_101))
    datautilus.printSingleDigit (feat_101, dataHouse.save, 'step10')

    #Step 11
    score = datautilus.batchEuclid (X_test, y_test, means_overall)
    print ('The trained Euclidean classifier performed ', score,
           'successfully on the test dataset.')

    #Step 13
    X_total = np.vstack((X_train, X_test))
    #print (y_train, y_test)
    y_total = np.concatenate((y_train, y_test))
    print ('Datasets joined!')
    lilEuclid = euclidean_classifier.EuclideanClassifier ()
    print ('Ready to cross_validate!')
    cv_results = cross_validate (lilEuclid, X_total, y_total, cv = 5)
    print ('CV-complete: ', cv_results)

"""
    #train_sizes = np.linspace((.1, 1.0, 5))
    train_sizes, train_scores, test_scores = learning_curve(lilEuclid,
                                                            X_total, y_total)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show ()
"""

if __name__ != '__main__':
    main ()
else:
    print ('Yo, this is not how we work. Better check tha README.')
