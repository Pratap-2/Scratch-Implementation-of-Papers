import numpy as np

r,c= map(int,input().split())
x = np.array([list(map(float,input().split())) for _ in range(r)])
y = np.array([float(input()) for _ in range(r)]).reshape(r,1)

num_classes = 3
w =np.random.randn(num_classes,c) * 0.01 
b =np.zeros((num_classes,1))


#one hot encoding 
Y= np.zeros((r,num_classes))
Y[np.arange(r),y]=1 # np.arange(r)-> these are the row indices
# so essentially [0,claslabel]=1 in matrix form


#axis=1 → operate across columns/classes for each sample.
#keepdims=True → keeps the reduced dimension as size 1.
#axis=0 → operate down the rows
#axis=0 → collapse rows → result for each COLUMN
#axis=1 → collapse columns → result for each ROW

def softmax(z):
    z = z - np.max(z , axis=1 ,keepdims=True)
    exp_z = np.exp(z)
    return exp_z/np.sum(exp_z,axis=1,keepdims=True)


def cross_entropy(y,ypred):
    return -np.sum(y * np.log(ypred),axis=1)


alpha = 0.01 
for epochs in range(100):
    z     = x @ w.T + b.T
    Ypred = softmax(z)
    loss  = cross_entropy(Y,Ypred)

    #backward
    error = Ypred - Y
    dw    = (error.T @ x) / r
    db    = np.mean(error,axis = 0 ,keepdims=True)

    #update
    w-= alpha * dw
    b-= alpha * db


# prediction
z = x @ w.T + b.T
Ypred = softmax(z)

classes = np.argmax(Ypred, axis=1)

print(classes)

actual_classes = np.argmax(Y, axis=1)
accuracy = np.mean(classes == actual_classes)
print(accuracy)