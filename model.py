import tensorflow as tf
import utils
import numpy as np
import cv2
from keras.applications.vgg19 import VGG19 #pre-trained 
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorboard import TensorBoard

numClasses = 2

def load_data_full(datadir, numClasses):
    data = []
    for i in range(0, numClasses):
        data.append(utils.loadData(datadir + "/" + str(i)))

    X = []
    y = []
    for idx, i in enumerate(data):
        label = idx
        imgs = i
        for img in imgs:
            X.append(cv2.resize(img, (224, 224))) #resize will resize the img to 224x224
            y.append(utils.conv_num_to_one_hot(label, numClasses))
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

#This is the input to the first CNN layer
#input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3]) #None is array size, height, width of the image, 3 is number of color channels RGB

#building model based on VGG19
base_model = VGG19(include_top=False, input_shape=(224, 224, 3))
model = Flatten(name='flatten')(base_model.output) #this will apply the flatten layer after the last output layer of VGG19
model = Dense(4096, activation='relu', name='fc1')(model) #This will apply the dense layer on top of the flatten layer
model = Dense(4096, activation='relu', name='fc2')(model) #This will apply the dense layer on top of the previous layer
model = Dense(numClasses, activation='softmax', name='outputs')(model) #This is for the final layer of size number of classes

model = Model(base_model.input, model) #This will take the nase model input and concatenate the model we have created

#Freezing the initial 16 layers so that it doesn't get used during training
for i in model.layers[:16]:
    i.trainable = False

#it sets the hyperparmaters 
model.compile(loss='categorical_crossentropy', optimizer='adam')
print("Model Created")

tfBoard = TensorBoard(log_dir="./logs")

X, y = load_data_full("./data", numClasses)
#Data augmentation to get more photos from existing photos
datagen = ImageDataGenerator(
    rotation_range=50,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest'
)
datagen.fit(X)

print("Starting Training")
model.fit_generator(datagen.flow(X, y, batch_size=32), steps_per_epoch=len(X)/32, epochs=20, callbacks=[tfBoard])
print("Saving Model")
model.save("model.h5")