import requests
import random 
import numpy as np
import pandas as pd
import time
def query(id):
    headers ={
      'Authorization': 'token 50ddbeae4b8e36e3b1e50ae7f1507e23b3ac7763', # replace <TOKEN> with your token
    }
    response = requests.get('https://api.github.com/users?since='+str(id),headers=headers)
    
    rst = response.json()
    return rst
def getQuery(id):
    while True:
        q = query(id)
        if len(q) != 30:
            print(q)
            print("sleep")
            time.sleep(600)
            print("wake")
        else:
            break
    return q


def estimate(path, budget):
    print(path+'\t'+str(budget))
    csvfile=open(path, 'w')
    count = 0
    i = 0
    while i < budget:
        id = int(random.uniform(0,MAX/100)) * 100
        q = getQuery(id)
        n = 0
        while q[-1]['id'] <= 100+id:
            n += 30
            nID = q[-1]['id']
            q = getQuery(nID)  
        for u in q:
            if u['id'] <= 100+id:
                n+=1
            else:
                break
        count += n
        if (i % 20 == 0):
            print(str(i)+":\t"+str(id)+','+str(n)+',\t'+str(count))
        csvfile.write(str(i)+','+str(id)+','+str(n)+'\n')
        i += 1
    result = int(count / budget) * MAX/100
    csvfile.close()
    return result

Budget = [100,200,300, 400,500]
MAX = 10870000
N = 10253550

rstfile = open('rst.csv', 'a')
rst = []
for b in Budget:
    p = 'estimate' + str(b)+'/estimate_'
    for i in range(50):
        path = p+ str(i)+'.csv'
        r = estimate(path, b)
        rstfile.write(str(b)+','+str(i)+','+str(r)+'\n')
        rst.append([b,i,r])
rstfile.close()
