import pickle

#open w2v


#make w2v to list or other


#make index
index = []
index_inv = {}
for i,word in words:
    index.append(word)
    index_inv[word] = i

#check
for i,word in enumerate(index_inv):
    if i!=index[word]:
        print('something is wrong')
        exit()

with open('data/index.pickle','wb') as f:
    pickle.dump(index,f)
with open('data/index_inv.pickle','wb') as f:
    pickle.dump(index_inv,f)
