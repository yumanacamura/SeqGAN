import pickle
import MeCab
import re
import numpy as np

shkigo = []
m = MeCab.Tagger('-Owakati')
p = re.compile(r'[（(].+?[）)]')

with open('data/index.pickle','rb') as f:
    index = pickle.load(f)
with open('data/index_inv.pickle','rb') as f:
    index_inv = pickle.load(f)

with open('data/hkigo.pickle','rb') as f:
    hkigo = pickle.load(f)

#parse haiku
for h,k in hkigo:
    rm = p.findall(h)
    for r in rm:
        h = h.replace(r,'')
    sh = m.parse(h).split()
    if len(sh)>19:
        print('yabai'+'#'*300,h)
        #input()
        continue
    if k not in sh:
        pass#print(k,'isnt in',sh)
        continue
    if False in [w in index for w in sh]:
        pass#print(''.join(sh),'isnt in index')
        continue
    shkigo.append((sh,k))


ihaiku = []
ikigo = []

#word2index
for sh,k in shkigo:
    for w in sh:
        ih = [index_inv[w] if w!=k else index_inv['季語'] for w in sh]
        ihaiku.append(ih + [0] + [1 for i in range(19-len(ih))])
        ikigo.append(index_inv[k])
with open('data/ihaiku_ikigo.pickle','wb') as f:
    pickle.dump((np.array(ihaiku),np.array(ikigo)),f)
