import pickle
import numpy as np
from keras.models import load_model
from keras.utils import np_utils

model_path = './kigo/models/model_20000_epechs.h5'
data_path = './result/result_1234_epoch.txt'
output_path = './result/output.txt'

model = load_model(model_path)

with open('../data/vec.pickle') as f:
    w2v = pickle.load(f)
with open('./data/index.pickle','rb') as f:
    index = pickle.load(f)

k = index.index('季語')

with open(data_path,'r') as f:
    haiku = [w2v[[int(word) for word in line.split(',')[0].split()]] for line in f.read().split('\n')]

out = open(output_path,'w')
for h in haiku:
    kigo = np.argmax(model.predict(h))
    out.write(''.join([index[word] if word!=k else index[kigo] for word in h])+'\n')
out.close()