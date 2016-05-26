
from pymongo import MongoClient, ReturnDocument
#f=open('pld.dat','w')

db = MongoClient('10.76.0.143', 27017)['plate']
db.authenticate('plateReader', 'IamReader')
#for post in db.image.find({'_id':'3'})
a=db.image.find_one({'_id':120762})
#f.write(str(a))
print a
b=db.image.find_one({'_id':524})
print b
c=db.image.find_one({'_id':1008})
print c
print db.image.count()
#f.close()

