import re
import pickle
import urllib.request
import urllib.error
from time import sleep
url = "http://www.haiku-data.jp/work_detail.php?cd="

ph = re.compile(r'(?<="><B>).+(?=&nbsp</B)')
pk = re.compile(r'(?<=季　語</font></div></td> \n                  <td width="76%" background="img/bg-w.gif" bgcolor="#FFFFFF"> <div align="center" class="font2">).+(?=&nbsp</div></td> \n                </tr> \n                <tr> \n                  <td background="img/bg6.gif"><div align="center"><span class="fontff"><font color="#FFFFFF">季　節)')

hkigo = []

for i in range(1,41360):
    sleep(0.2)
    resource = urllib.request.urlopen(url+str(i))
    contents = resource.read().decode()
    haiku = ph.findall(contents)
    kigo = pk.findall(contents)
    if not haiku or not kigo:
        print('empty at',i)
        continue
    hkigo.append(haiku+kigo)
with open('data/hkigo.pickle','wb') as f:
    pickle.dump(hkigo,f)
