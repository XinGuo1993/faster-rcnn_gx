# CREAT PHONE DATA
from pymongo import MongoClient, ReturnDocument
import os
import json
import shutil

db = MongoClient('10.76.0.143', 27017)['plate']
db.authenticate('plateReader', 'IamReader')
#f=open('plate/plate/Annotations/label.dat','r')
#f=open('phone_list','r') 
f1=open('phone_list.txt','w')

#for line in f:
    #esp=line
    #break
#print esp
name=os.listdir('/mnt/disk3/plate/src/cadphone')
for h,num in enumerate(name):
    #print esp
    #print h
    path=os.path.join('/mnt/disk3/plate/src/cadphone',num)
    image=db.image.find_one({'path':path})
    #print image['_id']
    #emp=json.loads(esp)
    #print temp['keypoint']
    #$or i in range(4):
         # print i
         # print type(i)
       # temp['keypoints'][i]=str(image['points'][0][i][0])+','+str(image['points'][0][i][1])
    #shutil.copy(image['path'],os.path.join('plate','phone','Images',str(h+1)+'.jpg'))
    #temp['path']=str(h+1)+'.jpg' 
    #json.dump(temp,f1)
    image['_id']=int(image['_id'])
    f1.write(str(image['_id']))
    f1.write('\n')
    #break
    
#f.close()
f1.close()







