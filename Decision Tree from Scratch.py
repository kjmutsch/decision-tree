# Databricks notebook source
# MAGIC %md
# MAGIC ## Decision Tree in Python from Scratch
# MAGIC
# MAGIC <p>This project is a passion project in my pursuit of machine learning knowledge. I felt to understand decision trees the best, I should make my own from scratch. I'm using the CART algorithm and gini impurity to build my tree.</p>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Training Data

# COMMAND ----------

# attributes: color, diameter, 
# label
training_data = [
  ['Green', 3, 'Apple'],
  ['Yellow', 4, 'Apple'],
  ['Red', 1, 'Grape'],
  ['Red', 1, 'Grape'],
  ['Yellow', 3, 'Lemon'],
]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Utility Functions
# MAGIC
# MAGIC <p>The general utility functions needed for a decision tree are as follows:</p>
# MAGIC 1. Gini Impurity Calculation for measuring the quality of a split
# MAGIC 2. Entropy/Information Gain Calculation to determine best split
# MAGIC 3. Data Splitting split dataset into features and theshold
# MAGIC 4. Tree Node Creation to create nodes and leaves
# MAGIC 5. Tree Building to recursively build the tree
# MAGIC 6. Prediction to make predicitons using the constructed tree
# MAGIC 7. Accuracy to test how good our model is on test data 

# COMMAND ----------


# Gini Impurity calculates the impurity or how likely it is that a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the set.
# y is the list of labels, in our small example it would be ['Apple', 'Apple', 'Grape', 'Grape', 'Lemon']
def gini_impurity(y):
  """Calculate the Gini Impurity for a list of class labels"""
  if not y or len(y) == 0:
    return 0.0
  from collections import Counter
  counts = Counter(y)
  # in our mini example it would look like Counter({Apple: 2, Grape: 2, Lemon: 1})
  impurity = 1
  total = len(y) # 3
  for label in counts:
    probability_of_label = counts[label] / total
    # for apple = 0.4
    # for grape = 0.4
    # for lemon = 0.2
    # then square them and subract from 1
    impurity -= probability_of_label ** 2
  return impurity

# COMMAND ----------

# Information Gain where parent_y is the parent node, left_y is the left child node, right_y is the right child node
def information_gain(parent_y, left_y, right_y):
  """Calculate the Information Gain of a split"""
  p = len(left_y) / len(parent_y)
  return gini_impurity(parent_y) - p * gini_impurity(left_y) - (1 - p) * gini_impurity(right_y)
  # returns the gini impurity of the parent (0.65) minus the probability of the left child (4/5) * the left child's gini impurity (0.625) minus the probability of the right child (1/5) * the right child's gini impurity (0)



# COMMAND ----------

# Data splitting where X are the features, y are the labels, feature_index is the index of the feature to split on, threshold is the value of the feature to split on
def split_dataset(X, y, feature_index, threshold):
  """Split the dataset based on a feature and threshold"""
  left_X, right_X = [],[]
  left_y, right_y = [],[]
  for i in range(len(X)):
    if X[i][feature_index] <= threshold:
      left_X.append(X[i])
      left_y.append(y[i])
    else:
      right_X.append(X[i])
      right_y.append(y[i])
  return left_X, right_X, left_y, right_y

# in our micro-example we'd have the left child with [['Green', 3], ['Red', 1], ['Red', 1], ['Yellow', 3]] with labels ['Apple', 'Grape', 'Grape', 'Lemon'] and the right child would have [['Yellow', 4]] with label(s) ['Apple'] and geni impurity after the fact would be 0.625 for the left and 0 for the right child

# COMMAND ----------

def find_best_split(X, y):
  """Find the best feature and threshold to split the data."""
  best_gain = 0
  best_feature_index = None
  best_threshold = None
  best_splits = None

  n_features = len(X[0]) # number of features

  for feature_index in range(n_features):
    thresholds = set([x[feature_index] for x in X])
    for threshold in thresholds:
      left_X, right_X, left_y, right_y = split_dataset(X, y, feature_index, threshold)
      if not left_X or not right_X:
        continue
      gain = information_gain(y, left_y, right_y)
      if gain > best_gain:
        best_gain = gain
        best_feature_index = feature_index
        best_threshold = threshold
        best_splits = (left_X, right_X, left_y, right_y)
  return best_feature_index, best_threshold, best_gain, best_splits

  # feature index and threshold are used when we split the data
  # best_gain is for testing and tracking effectiveness of split
  # best_splits is for when we build the tree
  

# COMMAND ----------

# Tree Node Creation
class TreeNode:
  def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
    self.feature_index = feature_index
    self.threshold = threshold
    self.left = left # left node
    self.right = right # right node
    self.value = value

# COMMAND ----------

# Tree Building
def build_tree(X, y, depth=0, max_depth=10):
  """Recursively build decision tree"""
  # if we reached the max depth or the node is the last in the set, then we set the leaf value as the most common value in the current set of labels
  if depth == max_depth or len(set(y)) == 1:
    leaf_value = max(set(y), key=list(y).count) # finds the element in the set y that has the highest count in the list y, key argument specifies a function to apply to each element before making a comparison. So we check what element has the most occurrences first (example 'Apple' could have 2)
    return TreeNode(value=leaf_value)
  
  # find the best split
  best_feature_index, best_threshold, best_gain, best_splits = find_best_split(X, y)

  # if no split, make it a leaf
  if best_gain == 0:
    leaf_value = max(set(y), key=list(y).count)
    return TreeNode(value=leaf_value)
  
  # create child nodes of this node
  left_X, right_X, left_y, right_y = best_splits
  left_child = build_tree(left_X, left_y, depth + 1, max_depth)
  right_child = build_tree(right_X, right_y, depth + 1, max_depth)

  return TreeNode(feature_index=best_feature_index, threshold=best_threshold, left=left_child, right=right_child)
  

# COMMAND ----------

def print_tree(node, depth=0):
  if node.value is not None:
    print(f"{'  ' * depth}Predict: {node.value}")
  else:
    print(f"{'  ' * depth}[X{node.feature_index} <= {node.threshold}]")
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)

# COMMAND ----------

# Test with our small example of data
X = [row[:2] for row in training_data]  # Features
y = [row[2] for row in training_data]   # Labels
# X are the features and they look like this:
""" [['Green', 3], ['Yellow', 4], ['Red', 1], ['Red', 1], ['Yellow', 3]] """
# y are the labels and they look like this:
""" ['Apple', 'Apple', 'Grape', 'Grape', 'Lemon'] """

# Build the tree
tree = build_tree(X, y, max_depth=3)

# Print the tree
print_tree(tree)

# COMMAND ----------

def accuracy(tree, X_test, y_test):
    """Calculate the accuracy of the decision tree on the test data."""
    correct_predictions = 0
    for i in range(len(X_test)):
        if predict(tree, X_test[i]) == y_test[i]:
            correct_predictions += 1
    return correct_predictions / len(X_test)

# COMMAND ----------

def predict(tree, X):
    """Make a prediction for a single instance X using the decision tree."""
    if tree.value is not None:
        return tree.value
    feature_value = X[tree.feature_index]
    print(feature_value, tree.threshold)
    if feature_value <= tree.threshold:
        return predict(tree.left, X)
    else:
        return predict(tree.right, X)

# COMMAND ----------

test_data = [
  ['Green', 3, 'Apple'],
  ['Yellow', 4, 'Apple'],
  ['Red', 1, 'Grape'],
  ['Yellow', 3, 'Lemon'],
]
X_test = [row[:2] for row in test_data]  # Features
y_test = [row[2] for row in test_data]   # Labels

# Make a prediction and test the accuracy of tree
predictions = [predict(tree, x) for x in X_test]
print("Predictions: ", predictions)

accur = accuracy(tree, X_test, y_test)
print("Accuracy:", accur)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use a the iris dataset to test model

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target # features and labels

print(X,y)

# COMMAND ----------

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # random state is so we can reproduce the same split, 20% of the data is used for testing

iris_tree = build_tree(X_train.tolist(), y_train.tolist(), max_depth=3)
print_tree(iris_tree)

# COMMAND ----------

# Evaluate the tree
# Calculate accuracy on the test data
acc = accuracy(iris_tree, X_test.tolist(), y_test.tolist())
print("Accuracy:", acc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tests
# MAGIC
# MAGIC #### Tests with simple dataset

# COMMAND ----------

def test_simple_dataset():
  # Simple dataset
  training_data = [
      [2.7, 2.5, 'A'],
      [1.3, 1.5, 'B'],
      [3.1, 3.0, 'A'],
      [1.1, 1.2, 'B'],
  ]
  
  X = [row[:2] for row in training_data]  # Features
  y = [row[2] for row in training_data]   # Labels
  
  # Build the tree
  tree = build_tree(X, y, max_depth=3)
  
  # Test predictions
  test_data = [
      [2.8, 2.6],  # Should be 'A'
      [1.2, 1.3],  # Should be 'B'
  ]
  
  predictions = [predict(tree, x) for x in test_data]
  expected = ['A', 'B']
  
  assert predictions == expected, f"Expected {expected}, but got {predictions}"
  print("test_simple_dataset passed")

test_simple_dataset()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Test with Iris Dataset

# COMMAND ----------

def test_iris_dataset():
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split

  # Load the iris dataset
  iris = load_iris()
  X, y = iris.data, iris.target

  # Split the dataset into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Build the tree using the training data
  tree = build_tree(X_train.tolist(), y_train.tolist(), max_depth=3)

  # Calculate accuracy on the test data
  acc = accuracy(tree, X_test.tolist(), y_test.tolist())
  
  assert acc > 0.7, f"Expected accuracy > 0.7, but got {acc}"
  print("test_iris_dataset passed")

test_iris_dataset()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Test of single class (all labeled 'A')

# COMMAND ----------

def test_edge_cases():
  # Dataset with only one class
  training_data = [
      [2.7, 2.5, 'A'],
      [3.1, 3.0, 'A'],
      [2.9, 2.8, 'A'],
  ]
  
  X = [row[:2] for row in training_data]  # Features
  y = [row[2] for row in training_data]   # Labels
  
  # Build the tree
  tree = build_tree(X, y, max_depth=3)
  
  # Test predictions
  test_data = [
      [2.8, 2.6],  # Should be 'A'
      [3.0, 2.9],  # Should be 'A'
  ]
  
  predictions = [predict(tree, x) for x in test_data]
  expected = ['A', 'A']
  
  assert predictions == expected, f"Expected {expected}, but got {predictions}"
  print("test_edge_cases passed")

test_edge_cases()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Test accuracy calculation

# COMMAND ----------

def test_accuracy_calculation():
  # Simple dataset
  training_data = [
      [2.7, 2.5, 'A'],
      [1.3, 1.5, 'B'],
      [3.1, 3.0, 'A'],
      [1.1, 1.2, 'B'],
  ]
  
  X = [row[:2] for row in training_data]  # Features
  y = [row[2] for row in training_data]   # Labels
  
  # Build the tree
  tree = build_tree(X, y, max_depth=3)
  
  # Test data
  test_data = [
      [2.8, 2.6, 'A'],
      [1.2, 1.3, 'B'],
      [3.0, 2.9, 'A'],
      [1.0, 1.1, 'B'],
  ]
  
  X_test = [row[:2] for row in test_data]  # Features
  y_test = [row[2] for row in test_data]   # Labels
  
  # Calculate accuracy
  acc = accuracy(tree, X_test, y_test)
  
  assert acc == 1.0, f"Expected accuracy 1.0, but got {acc}"
  print("test_accuracy_calculation passed")

test_accuracy_calculation()


# COMMAND ----------

def run_all_tests():
  test_simple_dataset()
  test_iris_dataset()
  test_edge_cases()
  test_accuracy_calculation()
  print("All tests passed")

run_all_tests()
