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

#which features mattered the most???(infer)
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature importances:\n", importances.sort_values(ascending=False))

#new sample
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=iris.feature_names
    )
    pred_class = model.predict(sample)[0]
    probs = model.predict_proba(sample)[0]  # confidence per class
 
    species_name = iris.target_names[pred_class]
    confidence = probs[pred_class]
 
    print(f"\nInput: sepal={sepal_length}x{sepal_width}, petal={petal_length}x{petal_width}")
    print(f"Predicted species: {species_name}  (confidence: {confidence:.1%})")
    for name, p in zip(iris.target_names, probs):
        print(f"   {name}: {p:.1%}")
    #new sample
predict_species(5.1, 3.5, 1.4, 0.2)   # classic setosa-shaped
predict_species(6.0, 2.7, 5.1, 1.6)   # more versicolor-shaped
predict_species(6.9, 3.1, 5.4, 2.1)  
return species_name
