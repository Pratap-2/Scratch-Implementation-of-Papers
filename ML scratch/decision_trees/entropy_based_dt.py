import numpy as np


class Node:
    def __init__(self,feature=None,threshold = None,left=None,right=None,prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

    def is_leaf(self):
        return self.prediction is not None


class DecisionTree:

    def __init__(
        self,max_depth=5,min_sample_split=2,min_sample_leaf=1,min_gain=0.01,max_features=None):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.min_gain = min_gain
        self.max_features = max_features

    def entropy(self,y):
        if len(y) == 0:
            return 0.0
        count={}
        for val in y:
            if val not in count:
                count[val]=0
            count[val]+=1

        entropy=0.0
        for data,val in count.items():
            p=val/len(y)
            entropy += -p*np.log2(p)

        return entropy

    def split(self,x,y,feature,threshold):
        if(len(y)==0):
            return 0

        parent_entropy = self.entropy(y)

        y1=[]
        y2=[]
        for row,label in zip(x,y):
            if(row[feature] <= threshold):
                y1.append(label)
            else:
                y2.append(label)

        entropy1 = self.entropy(y1)
        entropy2 = self.entropy(y2)

        weighted_entropy=(
            (len(y1)/len(y)) * entropy1 + 
            (len(y2)/len(y)) * entropy2
        )

        information_gain = parent_entropy - weighted_entropy

        return information_gain

    def split_dataset(self, x, y, feature, threshold):

        left_mask = x[:, feature] <= threshold

        right_mask = ~left_mask

        x_left = x[left_mask]
        y_left = y[left_mask]

        x_right = x[right_mask]
        y_right = y[right_mask]

        return x_left, y_left, x_right, y_right

    def find_best_split(self, x, y):

        n = x.shape[1]

        if self.max_features is None:
            candidate_features = np.arange(n)
        else:
            candidate_features = np.random.choice(
                n,
                size=min(self.max_features, n),
                replace=False
            )

        best_gain = -float("inf")
        best_threshold = None
        best_feature_split = None

        for i in candidate_features: #iterating over features sampled one

            feature_values = np.sort(x[:, i])
            m = len(feature_values)

            for j in range(1, m):

                # Skip duplicate adjacent values
                if feature_values[j] == feature_values[j - 1]:
                    continue

                temp_threshold = (
                    feature_values[j] + feature_values[j - 1]
                ) / 2

                temp_info_gain = self.split(
                    x, y, i, temp_threshold
                )

                if temp_info_gain > best_gain:
                    best_gain = temp_info_gain
                    best_threshold = temp_threshold
                    best_feature_split = i

        return best_gain, best_threshold, best_feature_split

    def find_pred(self,y):
        if len(y)==0:
            raise ValueError("Cannot predict from an empty label array")
        
        count={}
        for val in y:
            if val not in count.items():
                count[val]=0
            count[val]+=1
        
        majority_class=max(count,key=count.get)
        return majority_class

    def build_tree(self, x, y, depth):

        if len(y) == 0:
            raise ValueError("Cannot build a tree from empty data")

        # Stop if the node is pure.
        if len(np.unique(y)) == 1:
            return Node(prediction=y[0])

        # Stop if the node is too small.
        if len(y) < self.min_sample_split:
            return Node(prediction=self.find_pred(y))

        # Stop if maximum depth has been reached.
        if depth >= self.max_depth:
            return Node(prediction=self.find_pred(y))

        best_gain, best_threshold, best_feature_split = (
            self.find_best_split(x, y)
        )

        # Stop if no useful split exists.
        if (
            best_feature_split is None
            or best_threshold is None
            or best_gain <= self.min_gain
        ):
            return Node(prediction=self.find_pred(y))

        x1, y1, x2, y2 = self.split_dataset(
            x, y, best_feature_split, best_threshold
        )

        # Defensive check: both children must contain samples.
        if len(y1) == 0 or len(y2) == 0:
            return Node(prediction=self.find_pred(y))

        left = self.build_tree(x1, y1, depth + 1)
        right = self.build_tree(x2, y2, depth + 1)

        return Node(
            feature=best_feature_split,
            threshold=best_threshold,
            left=left,
            right=right
        )
    
    def fit(self,x,y):
        self.node= self.build_tree(x,y,depth=0)
        return self
    
    def predict_one(self,row,node):
        if(node.is_leaf()):
            return node.prediction
        
        if(row[node.feature] <=node.threshold):
            return self.predict_one(row,node.left)
        else:
            return self.predict_one(row,node.right)

    def predict(self,x):

        predictions=[]
        for row in x:
            prediction = self.predict_one(row,self.node)
            predictions.append(prediction)
        return np.array(predictions)

import numpy as np

X = np.array([
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 80],
    [5, 85],
    [6, 90]
])

y = np.array([0, 0, 0, 1, 1, 1])

tree = DecisionTree(
    max_depth=5,
    min_sample_split=2,
    min_sample_leaf=1,
    min_gain=0.01,
    max_features=None
)

tree.fit(X, y)

predictions = tree.predict(X)

print("Predictions:", predictions)
print("Actual:     ", y)
print("Accuracy:   ", np.mean(predictions == y))