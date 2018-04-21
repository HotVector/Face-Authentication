import tensorflow as tf
import utils
import numpy as np

def load_data_full(datadir, numClasses):
    data = []
    for i in range(0, numClasses):
        data.append(utils.loadData(datadir + "/" + str(i)))

    X = []
    y = []
    for idx, i in enuoii8merate(data):
        label = idx
        imgs = i
        for img in imgs:
            X.append(img)
            y.append(utils.conv_num_to_one_hot(label, numClasses))
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

inputTensor = tf.placeholder(tf.float32, [None, 299, 299, 1 ])
numClasses = 4

X, y = load_data_full("./data", numClasses)