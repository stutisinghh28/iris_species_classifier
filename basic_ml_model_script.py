import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#start by loading the data required
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # features 
y = pd.Series(iris.target, name="species") # labels 
#explore the data
print("Shape of features:", X.shape())
print("\nFirst 5 rows:\n", X.head())
print("\nSpecies classes:", list(iris.target_names))  
#split the data into test/train
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
