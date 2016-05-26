import json
import os
f=open('data/PLATE/gz_10w/ImageSets/trainval.txt','r')
f1=open('log.txt','a')
label_file = os.path.join('data/PLATE/gz_10w', 'Annotations', 'label.dat')
image_ids=[int(x.strip()) for x in f.readlines()]
#print image_ids
#for x in f.readlines():
   # print int(x.strip())
    #break
print min(image_ids)


print max(image_ids)

with open(label_file, 'r') as f:
    image_index = [json.loads(x.strip())['path'].encode('utf-8')
                    for ind, x in enumerate(f.readlines())
                     if ind in image_ids]
for i ,num in enumerate(image_index):
    print num
    if i==5:
        break
    
#f.close()
#f1.write()
f1.close()
f.close()

