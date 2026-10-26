import numpy as np

r,c= map(int,input().split())
x =np.array([list(map(float,input().split())) for _ in range(r)])
theta= np.array([float(input()) for _ in range(c)]).reshape(c,1)

y=np.array([float(input()) for _ in range(r)]).reshape(r,1)


def sigmoid(z):
    "returns sigmoid of the function"
    return 1 /(1+np.exp(-z))


epsilon = 1e-15
"clip is essentially saying: ->Computer, I know mathematically this probability"
" should be between 0 and 1, "
"but don't let floating-point calculations hand log() an exact 0."

alpha = 0.01

for _ in range(100):
    #linear combination
    z = x@theta

    #probability
    pred = sigmoid(z)

    #numerical safety
    pred=np.clip(pred,epsilon,1-epsilon)

    #error
    error = pred - y

    # for logistic regression we use binary cross entropy 
    loss = -np.mean((y*np.log(pred)) + (1-y)*np.log(1-pred))
    # loss  = np.mean(error**2)/2 -> invlaid it is for lr
    
    # derivatives
    gradient  = ((x.T) @ error ) / r #->calculated from loss function

    #update
    theta = theta - alpha *  gradient
    

print("theta:")
print(theta)

print("probabilities:")
print(sigmoid(x @ theta))

print("classes:")
print((sigmoid(x @ theta) >= 0.5).astype(int))