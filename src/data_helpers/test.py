from easydict import EasyDict as edict
x = edict()
y = {}

ids = range(100)
for i in ids:
    y[i] = {'a': 10, 'b': 20}

#print(type(y))
#print(y.keys())

x.aa = y
