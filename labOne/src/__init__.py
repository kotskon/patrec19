#Re-usability of the tool is a desired feature. Thus a (possibly
#redundantly complex) module-based scheme will be employed.
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
    [X_train, X_test, y_train, y_test] = datautilus.dataReader (
                                         dataHouse.train,
                                         dataHouse.test)
    print ('All aboard!')

    #Step 2
    print ('Picking pattern #131...')
    [feat_131, id_131] = datautilus.pickDigit (X_train, y_train, 131)
    print ('Done! Number is ', int (id_131))
    [step2_fig, axs] = plt.subplots ()
    [axs, im] = datautilus.digit2Fig (feat_131, axs)
    step2_fig.colorbar (im)
    axs.set_ylabel ('pixel no.')
    axs.set_xlabel ('pixel no.')
    step2_fig.savefig(dataHouse.save + 'step2.svg')
    #plt.show ()
    plt.close ()

    #Step 3
    print ('Commencing serial digit search...')
    [posx, posy] = [0, 0]
    [step3_fig, axs] = plt.subplots (2, 5, figsize = (20, 20))
    for num in range(10):
        step3_num = datautilus.findDigit (X_train, y_train, num).iloc[0, :]
        [axs[posx, posy], im] = datautilus.digit2Fig (step3_num,
                                                      axs[posx, posy])
        posy += 1
        if posy % 5 == 0:
            posy = 0
            posx += 1

    step3_fig.colorbar (im, ax = axs, orientation = 'horizontal',
                        fraction = .025)
    step3_fig.savefig(dataHouse.save + 'step3.svg')
    #plt.show ()
    plt.close ()

    #Step 4
    print ('Collecting all instances of the digit 0...')
    step4_feats = datautilus.findDigit (X_train, y_train, 0)
    pix_1010_0 = datautilus.pixelSummoner (step4_feats, (10, 10))
    print ('Mean value for 0 pixel (10, 10) is: ', pix_1010_0.mean ())

    #Step 5
    print ('Variance for 0 pixel (10, 10) is: ', pix_1010_0.var ())

    #Steps 6 - 8
    mean_train_0 = []
    var_train_0 = []
    for i in range (16):
        i += 1
        for j in range (16):
            j += 1
            next_pix = datautilus.pixelSummoner (step4_feats, (i, j))
            mean_train_0.append (next_pix.mean ())
            var_train_0.append (next_pix.var ())
    print ('Mean, var data for all pixels of 0 collected! Saving results...')
    [steps6_8_fig, axs] = plt.subplots (1, 2, squeeze = False, figsize = (20, 20))
    [axs[0, 0], im] = datautilus.digit2Fig (pd.Series (mean_train_0),
                                                       axs[0, 0])
    steps6_8_fig.colorbar (im, ax = axs[0, 0], fraction = .04)
    [axs[0, 1], im] = datautilus.digit2Fig (pd.Series (var_train_0),
                                                       axs[0, 1])
    steps6_8_fig.colorbar (im, ax = axs[0, 1], fraction = .04)
    steps6_8_fig.savefig(dataHouse.save + 'steps6_8.svg')
    #plt.show ()
    plt.close ()

if __name__ != '__main__':
    main ()
else:
    print ('Yo, this is not how we work. Better check tha README.')
