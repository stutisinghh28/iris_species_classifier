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
#train a model(the real deal)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#evaluate the trained model
y_pred = model.predict(X_test)
 
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc:.2%}")
 
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
 
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))
