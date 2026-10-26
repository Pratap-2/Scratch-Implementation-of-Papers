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

    max_depth = 5
    min_sample_split = 2
    min_sample_leaf = 1
    min_gain = 0.01
    
    def gini_impurity(self,y):
        count={}
        for val in y:
            if val not in count:
                count[val]=0
            count[val]+=1

        n =len(y)
        gini=1.0
        for type,val in count.items():
            p = val / n 
            gini -= (p**2)
        return gini

    def split_dataset(self, x, y,threshold,feature):
        x1=[]
        x2=[]
        y1=[]
        y2=[] 
        for row,label in zip(x,y):
            if(row[feature]<=threshold):
                x1.append(row)
                y1.append(label)
            else:
                x2.append(row)
                y2.append(label)
        return np.array(x1), np.array(y1), np.array(x2), np.array(y2)
    
    def split(self,x,y,threshold,feature):
        if(len(y)==0): return 0
        parent_gini = self.gini_impurity(y)

        y1=[]
        y2=[]
        for row,label in zip(x,y): # u push all row in x1
            if(row[feature]<=threshold):
                y1.append(label)
            else:
                y2.append(label)

        gini1 = self.gini_impurity(y1)
        gini2 = self.gini_impurity(y2)

        netgini = ((len(y1)/len(y))*gini1 + (len(y2)/len(y))*gini2)

        gini_gain = parent_gini-netgini

        return gini_gain

    def finding_best_split(self,x,y):

        n = x.shape[1] # did mistake here shape(0) is wrong

        allfeaturesplitbest = []
        for i in range(n):#pick one feature
            feature_values=x[:,i] #careful for ith row pich whole row
            m= len(feature_values)
            # IMPORTANT: np.sort returns a new array
            feature_values = np.sort(x[:, i])
            count = []
            
            for j in range(1,m):
                temp_threshold = (feature_values[j-1]+feature_values[j]) / 2 
                temp_gini_gain = self.split(x,y,temp_threshold,i)
                count.append([temp_threshold,temp_gini_gain])
            max_gain = -float("inf")
            best_threshold = None

            for threshold,gain in count:
                if(gain>max_gain):
                    max_gain=gain
                    best_threshold = threshold

            allfeaturesplitbest.append((i,max_gain,best_threshold))

        best_feature_split= None
        best_gain =-float("inf")
        best_threshold = None

        for feature,gain,threshold in allfeaturesplitbest:
            if(gain>best_gain):
                best_gain = gain
                best_threshold = threshold
                best_feature_split = feature

        return best_gain,best_threshold,best_feature_split

    def find_pred(self,y):
        count={}
        for val in y:
            if val not in count:
                count[val]=0
            count[val]+=1

        maxi=-1
        majority_class=-1
        for type,val in count.items():
            if(val>maxi):
                maxi=val
                majority_class=type

        return majority_class
    
    def build_tree(self,x,y,depth):
        if len(y) < self.min_sample_split:
            return Node(prediction=self.find_pred(y))
        
        best_gain,best_threshold,best_feature_split = self.finding_best_split(x,y)
        
        if(best_gain<=self.min_gain or depth>=self.max_depth):#leaf
            return Node(prediction=self.find_pred(y))
        
        x1, y1, x2, y2 = self.split_dataset(x, y, best_threshold, best_feature_split)
        left  = self.build_tree(x1,y1,depth+1)
        right = self.build_tree(x2,y2,depth+1)

        return Node(best_feature_split,best_threshold,left,right)

    def fit(self,x,y):
        self.root = self.build_tree(x,y,0)
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
            prediction = self.predict_one(row,self.root)
            predictions.append(prediction)
        return np.array(predictions)


X = np.array([
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 80],
    [5, 85],
    [6, 90]
])

y = np.array([0, 0, 0, 1, 1, 1])
tree = DecisionTree()
print(tree.finding_best_split(X, y))

#model training 
tree.fit(X, y)
print(tree.predict(X))

#model testing 
X_test = np.array([
    [1.5, 62],
    [3.2, 72],
    [4.5, 83],
    [6.5, 95]
])

print(tree.predict(X_test))