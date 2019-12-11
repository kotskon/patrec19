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
    digIdx = valText.find ('1' or '2' or '3' or '4' or '5' or '6'           \
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

def featFactory (wavs, rates, win_sec = 25e-3, ov_sec = 10e-3,              \
                 calc_deltas = True, mfsc = False):
    """
    This function extracts features from wavs. Rates are used to ensure proper
    calibration of the window and overlap parameters. By default, MFCC with
    deltas are calculated.
    """
    feats = []
    deltas = []
    deltas2 = []
    for i in range (len (wavs)):
        win = round (win_sec * rates[i])
        hop = round ((win_sec - ov_sec) * rates[i])
        if mfsc == False:
            feats.append (librosa.feature.mfcc (wavs[i], sr = rates[i],     \
                          #by default, librosa returns features as rows. We take
                          #the transpose to have them as columns.
                          n_mfcc = 13, win_length = win, hop_length = hop).T)
        else:             #melspectrogram is used to bypass the DCT performed by
                          #mfcc
            feats.append (librosa.feature.melspectrogram (wavs[i], sr = rates[i],
                          win_length = win, hop_length = hop, n_mels = 13).T)
        if calc_deltas:
            deltas.append (librosa.feature.delta (feats[i].T).T)
            deltas2.append (librosa.feature.delta (feats[i].T, order = 2).T)
    return feats, deltas, deltas2

def histPlotter (n1, feature, feats, digits, speakers):
    """
    This function draws the histograms of a certain feature of
    a certain digit across all speakers.
    """
    fig = plt.figure (figsize = (10, 10))
    fig_idx = 1
    sigslice = [feats[i][:, feature - 1] for i in range (len (feats)) if    \
                digits[i] == n1]
    speakslice = speakers[digits == n1]
    for i in range (len (sigslice)):
        ax = fig.add_subplot (4, 4, fig_idx)
        ax.hist (sigslice[i])
        ax.set_title ('Speaker ' + str (speakslice[i]))
        fig_idx += 1
    fig.tight_layout ()
    plt.show ()

def featCompression (feats, deltas, deltas2):
    """
    Returns augmented feature vectors for all cases.
    """
    feats_total = np.zeros (78)
    for i in range (len (feats)):
        row_total = np.array ([])
        feat_mean = np.mean (np.array (feats[i]), axis = 0)
        delt_mean = np.mean (np.array (deltas[i]), axis = 0)
        delt2_mean = np.mean (np.array (deltas2[i]), axis = 0)
        feat_std = np.std (np.array (feats[i]), axis = 0)
        delt_std = np.std (np.array (deltas[i]), axis = 0)
        delt2_std = np.std (np.array (deltas2[i]), axis = 0)
        row_total = np.hstack ((feat_mean, feat_std, delt_mean, delt_std, \
                                delt2_mean, delt2_std))
        feats_total = np.vstack ((feats_total, row_total))
    return feats_total[1:, :]
