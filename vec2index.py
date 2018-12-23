import pickle
import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors

model = Word2VecKeyedVectors.load('model/wv.model')

index = model.index2entity
index_inv = {}
for i,word in enumerate(index):
    index_inv[word] = i+1
index_inv['<p>'] = 0
with open('data/index.pickle','wb') as f:
    pickle.dump(['<p>']+index,f)
with open('data/index_inv.pickle','wb') as f:
    pickle.dump(index_inv,f)
z = np.zeros(model.wv[index[0]].shape,dtype='float32')
with open('data/vec.pickle','wb') as f:
    pickle.dump(np.array([z]+[model.wv[word] for word in index]),f)
