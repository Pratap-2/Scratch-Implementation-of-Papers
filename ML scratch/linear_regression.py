import numpy as np
x=np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

w=0
b=0
# pred=  w*x+b
# print(pred)


# error = pred - y
# loss = np.mean(error*2)
# print(loss)

l_rate = 0.01
# dw = 2*np.sum(x * (pred-y))/len(x)
# db = 2*np.sum(pred-y)/len(x)

# w= w-l_rate*dw
# b= b-l_rate*db


for epoch in range(10):
    pred = w*x+b
    error = pred-y 
    loss = np.mean(error**2)

    dw = (2/len(x)) * (np.sum(x * (error)))
    db = (2/len(x)) * (np.sum(error))

    w = w - l_rate*dw 
    b = b - l_rate*db

print(w*x+b)