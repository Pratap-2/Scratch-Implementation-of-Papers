import numpy as np

def gini_impurity(y):
    count = {}

    for value in y:
        if value not in count:
            count[value] = 0
        count[value] += 1

    n = len(y)
    gini = 1.0

    for type, val in count.items():
        p = val / n
        gini -= p ** 2

    return gini


def split_dataset(x,y,feature_index,threshold):
    
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    for a,b in zip(x,y):
        if(a[feature_index]<=threshold):
            x1.append(a)
            y1.append(b)
        else:
            x2.append(a)
            y2.append(b)

    return x1,y1,x2,y2

X = np.array([
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 80],
    [5, 85],
    [6, 90]
])

y = np.array([0, 0, 0, 1, 1, 1])

x1, y1, x2, y2 = split_dataset(X, y, 0, 3.5)
# print(x1,y1)
# print(x2,y2)

def best_split(x, y, feature):

    parent_gini = gini_impurity(y)
    bsplit = {}
    prev = 0
    cnt = 0

    for val in x[:, feature]:

        if cnt == 0:
            prev = val
            cnt += 1
            continue

        cnt += 1

        mid = (val + prev) / 2

        y1 = []
        y2 = []

        for z1, z2 in zip(x, y):

            if z1[feature] <= mid:
                y1.append(z2)
            else:
                y2.append(z2)

        gini1 = gini_impurity(y1)
        gini2 = gini_impurity(y2)

        n = len(y)

        netgini = parent_gini - (
            (len(y1) / n) * gini1 +
            (len(y2) / n) * gini2
        )

        bsplit[mid] = netgini

        prev = val

    best_gain = -1
    best_split_value = None

    for split, gain in bsplit.items():

        if gain > best_gain:
            best_gain = gain
            best_split_value = split
    print(best_split_value)
    print(best_gain)
    return best_split_value, best_gain



def best_split1(x, y):

    n_features = x.shape[1]

    feature_split = []

    for feature in range(n_features):

        best_split_value, best_gain = best_split(
            x,
            y,
            feature
        )

        feature_split.append(
            (feature, best_split_value, best_gain)
        )

    # Find the feature/threshold with maximum gain

    best_feature = None
    best_threshold = None
    best_gain = -1

    for feature, threshold, gain in feature_split:

        if gain > best_gain:

            best_gain = gain
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold, best_gain



X = np.array([
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 80],
    [5, 85],
    [6, 90]
])

y = np.array([0, 0, 0, 1, 1, 1])

feature, threshold, gain = best_split1(X, y)

print(feature)
print(threshold)
print(gain)