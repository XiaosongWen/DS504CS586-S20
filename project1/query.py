import requests
def query(id):
    headers ={
      'Authorization': 'token 1f4ad6f2d3263f275715bbce9d9a9eec20d19c04', # replace <TOKEN> with your token
    }
    response = requests.get('https://api.github.com/users?since='+str(id),headers=headers)
    
    rst = response.json()
    return rst


id = 330069
count = 300000
import numpy as np
import pandas as pd
import csv
import time
index = 0
csvfile=open('dataTest.csv', 'a')
# start = time.time()
fn = 'e'
print(111111111111111111)
while id < 649556:
  data = query(id)  
  if(len(data)==30):
    index += 1
    count += len(data)
    id = data[-1]['id']
    csvfile.write(str(id)+','+str(count)+'\n')
    if(index % 100 == 0):
      print(str(index)+':\t'+str(id)+','+str(count)) 
  else:
    print(data)
    print(index)
    
    end = time.time()
    # running_time = end-start
    # print('Rest: \t'+str(4000-running_time))
    time.sleep(600)
    # start = time.time()
    # time.sleep()    
    print("start")