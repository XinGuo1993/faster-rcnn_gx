#import os
f=open('predict_result.txt','r')
f1=open('faster_rcnn_detect.txt','w')
#line=f.readline()
for line in f:
    num=0
    #print len(line)
    if len(line)>10:
        for i in range(len(line)):
            if line[i]==' ':
                num=num+1            
        if num<12:
            a=line.split()
            f1.write(line[0:len(line)-1])
            f1.write('*'+' '+'None')
            f1.write('\n')
        else:
            f1.write(line)
f.close()    
f1.close()
