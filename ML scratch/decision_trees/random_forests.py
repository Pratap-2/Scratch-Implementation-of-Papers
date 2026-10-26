import numpy as np
x=[1,2,3,4,5,6,7]
y = np.random.randint(0,len(x),size=len(x))  #used for bootstrap sampling 
z = np.random.choice(x,size=3,replace=False) #used for feature sampling
print(y)
print(z) 


# | Operation                             | NumPy function                               | Replacement         | Duplicates allowed? |
# |
# | Bootstrap sampling of rows            | `np.random.randint(0, len(x), size=len(x))`  | With replacement    | Yes |
# | Random selection of distinct features | `np.random.choice(n, size=k, replace=False)` | Without replacement | No |


from entropy_based_dt import *

class RandomForest:
    def __init__(self,n_estimators=10,max_depth=5,
                 min_sample_split=2,min_sample_leaf=1,
                 min_gain=0.01,max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.min_gain = min_gain
        self.max_features = max_features

        self.trees = []

    def bootstrap_sample(self,x,y):
        n_samples = x.shape[0]
        indices   = np.random.randint(0,n_samples,size=n_samples)
        x_sample  = x[indices]
        y_sample  = y[indices]

        return x_sample,y_sample

    def fit(self,x,y):
        x=np.asarray(x)
        y=np.asarray(y)

        n_features = x.shape[1]

        #determine the number of candidate features
        if self.max_features == "sqrt":
            max_features = max(1,int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            max_features = max(1,int(np.log2(n_features)))
        elif self.max_features is None:
            max_features = None
        else:
            max_features = int(self.max_features)
            if max_features <1:
                raise ValueError("max_features must be atleast 1")
            max_features = min(max_features,n_features)

        self.trees = []
        for _ in range(self.n_estimators):
            #1. create a bootstrap sample
            x_sample , y_sample = self.bootstrap_sample(x,y)
            #2. Create a decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_sample_split=self.min_sample_split,
                min_sample_leaf=self.min_sample_leaf,
                min_gain=self.min_gain,
                max_features=max_features
            )

            #3. train the tree
            tree.fit(x_sample,y_sample)

            #4. store the trained tree
            self.trees.append(tree)

        return self

    #3. majority voting
    def predict(self,x):

        x=np.asarray(x)
        # Predictions from every tree
        all_predictions = np.array([
            tree.predict(x) 
            for tree in self.trees
        ])

        # Shape: (n_estimators, n_samples)
        final_predictions = []
        for sample_predictions in all_predictions.T:
            counts={}
            for pred in sample_predictions:
                if pred not in counts:
                    counts[pred] = 0

                counts[pred] += 1

            # Select the class with the most votes
            majority_class = max(counts,key=counts.get())
            final_predictions.append(majority_class)
        return np.array(final_predictions)


    

  
import numpy as np
from random_forests import RandomForest

# Reproducible dataset
np.random.seed(42)

n_samples = 1000

# Features:
# 1. Annual income (in thousands)
income = np.random.normal(60, 20, n_samples)

# 2. Debt-to-income ratio (0 to 1)
debt_ratio = np.random.uniform(0, 1, n_samples)

# 3. Credit score
credit_score = np.random.normal(680, 60, n_samples)

# 4. Number of missed payments
missed_payments = np.random.poisson(1.5, n_samples)

# Combine features into a matrix
X = np.column_stack([
    income,
    debt_ratio,
    credit_score,
    missed_payments
])

# Create a risk score
risk_score = (
    -0.025 * (income - 60)
    + 3.0 * debt_ratio
    - 0.012 * (credit_score - 680)
    + 0.8 * missed_payments
)

# Convert risk scores into binary labels
probability = 1 / (1 + np.exp(-risk_score))

# Generate labels with randomness
y = (np.random.random(n_samples) < probability).astype(int)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Class counts:", np.bincount(y))
