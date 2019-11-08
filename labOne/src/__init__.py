#Re-usability of the tool is a desired feature. Thus a (possibly
#redundantly complex) module-based scheme will be employed.
import config
from src import datautilus
import matplotlib.pyplot as plt

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
    plt.close ()

    #Step 3
    print ('Commencing serial digit search...')
    [posx, posy] = [0, 0]
    [step3_fig, axs] = plt.subplots (2, 5, figsize = (20, 20))
    for num in range(10):
        step3_num = datautilus.findDigit (X_train, y_train, num)
        [axs[posx, posy], im] = datautilus.digit2Fig (step3_num,
                                                      axs[posx, posy])
        posy += 1
        if posy % 5 == 0:
            posy = 0
            posx += 1

    step3_fig.colorbar (im, ax = axs, orientation = 'horizontal',
                        fraction = .025)
    step3_fig.savefig(dataHouse.save + 'step3.svg')
    plt.close ()

if __name__ != '__main__':
    main ()
else:
    print ('Yo, this is not how we work. Better check tha README.')
