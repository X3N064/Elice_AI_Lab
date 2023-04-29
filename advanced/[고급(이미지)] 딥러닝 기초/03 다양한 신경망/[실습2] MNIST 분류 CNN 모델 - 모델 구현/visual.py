import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from elice_utils import EliceUtils
elice_utils = EliceUtils()

def Visulaize(histories, key='loss'):
    for name, history in histories:
        plt.plot(history.epoch, history.history[key], 
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])    
    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")