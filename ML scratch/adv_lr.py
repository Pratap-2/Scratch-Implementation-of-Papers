import numpy as np
#go into the advance application of lr in matrix format

r , c= map(int ,input().split())
x=np.array([list(map(float,input().split())) for _ in range(r)])

theta = np.array([float(input()) for _ in range(c)]).reshape(c,1)

print(x)
print(theta)

y = np.array([float(input()) for _ in range(r)]).reshape(r,1)


for i in range(1000):
    pred = x@theta
    error= pred -y

    loss = np.mean(error**2)/2


    #derivatives for the same 

    gradient  = (x.T @ error) / r

    # print(gradient)
    alpha = 0.01

    theta = theta - alpha * gradient 

    if i % 100 ==0:
        print("iteration:", i)
        print("loss:",loss)
        print("theta:")
        print(theta)


print("Final resultas are :->")
print(x @ theta)

























