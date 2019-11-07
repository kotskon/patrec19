#Re-usability of the tool is a desired feature. Thus a (possibly
#redundantly complex) module-based scheme will be employed.
import config
from src import datautilus

#A global variable.
dataHouse = config.ToolConfig ()

def main ():
    """
    This function works as a general orchestrator of the tool.
    """
    print ('Module initiated. Reading data...')
    #Step 1
    [X_train, X_test, y_train, y_test] = datautilus.dataReader (
                                         dataHouse.train,
                                         dataHouse.test)
    print ('All aboard!')
    #Step 2
    digit_131 = X_train.iloc[131, :]
    #print (digit_131)
    datautilus.printDigit (digit_131)

if __name__ != '__main__':
    main ()
else:
    print ('Yo, this is not how we work. Better check tha README.')
