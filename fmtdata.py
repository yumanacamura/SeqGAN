import pickle
import MeCab
import re
from collections import Counter


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
        input()
        continue
    if k not in sh:
        print(k,'isnt in',sh)
        continue
    if False in [w in index for w in sh]:
        print(''.join(sh),'isnt in index')
    words+=sh
    shkigo.append((sh,k))

#make index
index = ['<EOS>','<p>']+words
index_inv = {w:i for i,w in enumerate(index)}


ihaiku = []
ikigo = []

#word2index
for sh,k in shkigo:
    for w in sh:
        if w not in words:
            print(w,'is too little')
            continue
        ih = [index_inv[w] for w in sh]
        ihaiku.append(ih + [0] + [1 for i in range(19-len(ih))])
        ikigo.append(index_inv[k])
with open('data/ihaiku_ikigo.pickle','wb') as f:
    pickle.dump((np.array(ihaiku),np.array(ikigo)),f)
with open('data/index.pickle','wb') as f:
    pickle.dump(index,f)
with open('data/index_inv.pickle','wb') as f:
    pickle.dump(index_inv,f)
