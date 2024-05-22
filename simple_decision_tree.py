#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

#dataset
from sklearn.datasets import load_wine

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name = 'class')

X.head()

#data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training samples: {X_train.shape[0]}')
print(f'Test samples: {X_test.shape[0]}')

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy: .2f}')

report = classification_report(y_test, y_pred, target_names=data.target_names)
print('Classification Report:\n', report)

con_mat = confusion_matrix(y_test, y_pred)
sn.heatmap(con_mat, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('predicted')
plt.ylabel('real')
plt.title('confusion matrix')
plt.show()

#visualize the tree
plt.figure(figsize=(10,8))#adjust the values for readability
plot_tree(classifier, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
