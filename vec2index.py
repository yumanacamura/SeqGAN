import pickle
from gensim.models.keyedvectors import Word2VecKeyedVectors

model = Word2VecKeyedVectors.load('model/wv.model')

#make index
index = model.index2entity
index_inv = {}
for i,word in enumerate(index):
    index_inv[word] = i

#check
for i,word in enumerate(index):
    if i!=index_inv[word]:
        print('something is wrong')
        exit()

with open('data/index.pickle','wb') as f:
    pickle.dump(index,f)
with open('data/index_inv.pickle','wb') as f:
    pickle.dump(index_inv,f)
