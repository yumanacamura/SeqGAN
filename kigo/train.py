import pickle
import numpy as np

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, Dropout, BatchNormalization, ZeroPadding2D, Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D
from keras.layers.merge import Concatenate
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.utils import np_utils

np.random.seed(0)
np.random.RandomState(0)
tf.set_random_seed(0)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

iterations = 20000
batch_size = 8192
test_interval = 200
save_interval = 2000

FILENAME = './model/predict.model'

#load data
with open('../data/vec.pickle') as f:
    w2v = pickle.load(f)
with open('../data/ihaikuikigo.pickle') as f:
    ihaiku,ikigo = pickle.load(f)

vocab_size,w2v_size = w2v.shape

h_input = Input((20,w2v_size))

model = Conv1D(32, kernel_size=3, strides=2, padding="same")(i_input)
model = LeakyReLU(alpha=0.2)(model)
model = Dropout(0.25)(model)
model = Conv1D(64, kernel_size=3, strides=2, padding="same")(model)
model = LeakyReLU(alpha=0.2)(model)
model = Dropout(0.25)(model)
model = BatchNormalization(momentum=0.8)(model)
model = Conv1D(128, kernel_size=3, strides=2, padding="same")(model)
model = LeakyReLU(alpha=0.2)(model)
model = Dropout(0.25)(model)
model = BatchNormalization(momentum=0.8)(model)
model = Conv1D(256, kernel_size=3, strides=1, padding="same")(model)
model = LeakyReLU(alpha=0.2)(model)
model = Dropout(0.25)(model)

model = Flatten()(model)
model = Dense(vocab_size, activation='softmax')(model)

predict_kigo = Model(inputs=h_input, outputs=model)

predict_kigo.summary()

predict_kigo.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

result = open('result.txt','w')
test_haiku = np.array([w2v[h] for h in ihaiku[range(1000)]])
test_kigo = np.utils.to_categorical(ikigo[range(1000)],vocab_size)
for iteration in range(iterations):
    idx = np.random.randint(1000, ihaiku.shape[0], batch_size)
    haiku = np.array([w2v[h] for h in ihaiku[idx]])
    kigo = np.utils.to_categorical(ikigo[idx],vocab_size)

    loss = predict_kigo.train_on_batch(haiku,kigo)

    print("%d loss: %f, acc.: %.2f%%" % (iteration, loss[0], 100 * loss[1]))
    if (iteration+1) % test_interval == 0:
        score = model.ecvaluate(test_haiku,test_kigo)
        print("%d loss: %f, acc.: %.2f%%" % (iteration, score[0], 100 * score[1]))
        result.write("%d loss: %f, acc.: %.2f%%" % (iteration, score[0], 100 * score[1]))
    if (iteration+1) % model_interval == 0:
        predict_kigo.save("model_{0}_epechs.h5")

result.close()
