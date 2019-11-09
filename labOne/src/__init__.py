import config
from src import datautilus
import matplotlib.pyplot as plt
import pandas as pd

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
    means_overall = pd.DataFrame ()
    vars_overall = pd.DataFrame ()
    for num in range(10):
        #Step 3
        step3_num = datautilus.findDigit (X_train, y_train, num).iloc[0, :]
        #Steps 6-9
        mean_num, var_num = datautilus.analyzeDigit (X_train, y_train, num,
                                                     dataHouse.save)
        means_overall = means_overall.append (mean_num, ignore_index = True)
        vars_overall = vars_overall.append (var_num, ignore_index = True)
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
           means_overall.iloc[0, :].values[flat_idx])
    print ('For digit 0, variance of pixel (10, 10) is ',
           vars_overall.iloc[0, :].values[flat_idx])

    #Step 10
    feat_101, id_101 = datautilus.pickDigit (X_train, y_train, 101)
    pred_101 = datautilus.askEuclid (means_overall, feat_101)
    print ('prediction: ', pred_101)
    print ('actual: ', int (id_101))
    datautilus.printSingleDigit (feat_101, dataHouse.save, 'step10')

    #Step 11
    score, pairs = datautilus.batchEuclid (X_test, y_test, means_overall)
    print ('The trained Euclidean classifier performed ', score,
           'successfully on the test dataset.')

if __name__ != '__main__':
    main ()
else:
    print ('Yo, this is not how we work. Better check tha README.')
