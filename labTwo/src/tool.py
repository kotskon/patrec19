import librosa
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt

def nameToFacts (fileName):
    """
    This function returns in int form the parsed results of sound
    files named like 'six2.wav'.
    """
    valText = fileName[:fileName.find ('.')]
    digIdx = valText.find ('1' or '2' or '3' or '4' or '5' or '6' \
                           or '7' or '8' or '9')
    digString = valText[:digIdx]
    speakNum = int (valText[digIdx:])
    return speakNum, {
                        'one' :     1,
                        'two' :     2,
                        'three' :   3,
                        'four' :    4,
                        'five' :    5,
                        'six' :     6,
                        'seven' :   7,
                        'eight' :   8,
                        'nine' :    9
                     }.get (digString)

def dataParser (path):
    """
    This function parses all files in the directory specified by
    the path input variable. It is assumed that all said files are
    valid .wav files.
    """
    waves = []
    rates = []
    digits = []
    speakers = []
    files = [f for f in listdir (path) if isfile (join (path, f))]
    for i in range (len (files)):
        #Keep both the signals themselves, and the sampling rates.
        sig, rate = librosa.load (join (path, files[i]), sr = None)
        waves.append (sig)
        rates.append (rate)
        jspeak, jdig = nameToFacts (files [i])
        digits.append (jdig)
        speakers.append (jspeak)
    print ('Parsing complete! ', len (waves), ' files in total.')
    return waves, np.array (digits), np.array (speakers), rates

def histPlotter (n1, feature, feats, digits, speakers):
    """
    This function draws the histograms of a certain feature of
    a certain digit across all speakers.
    """
    fig = plt.figure (figsize = (10, 10))
    fig_idx = 1
    for i in range (digits.size):
        if digits[i] == n1:
            ax = fig.add_subplot (4, 4, fig_idx)
            ax.hist (feats[i][feature - 1, :])
            ax.set_title ('Speaker ' + str (speakers[i]))
            fig_idx += 1
    fig.tight_layout ()
    plt.show ()
