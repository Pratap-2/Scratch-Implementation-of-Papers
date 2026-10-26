import numpy as np

def mse(self,y):
    mean=0
    for value in y:
        mean+=value
    mean/=len(y)
    error =0
    for val in y:
        error+=(val-mean)**2
    return error/len(y)

def split(self,x,y,feature,threshold):
    parent_mse = mse(y)
    y1=[]
    y2=[]
    for row,label in zip(x,y):
        if row[feature]<=threshold:
            y1.append(label)
        else:
            y2.append(label)

    mse1 = mse(y1)
    mse2 = mse(y2)

    weighted_mse = (
        (len(y1)/len(y)) * mse1 +
        (len(y2)/len(y)) * mse2
    )

    net_mse = parent_mse - weighted_mse

    return net_mse

def find_pred(self,y):
    total=0
    for val in y:
        total+=val
    return total/len(y) # there is no majority class just predict the mean
#as mean minimizes the error


