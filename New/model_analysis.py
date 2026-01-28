# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:00:16 2026

@author: taylo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import KNeighborsClassifier

import joblib



#read in data set
df = pd.read_csv("hazards_10k.csv")

#splitting into different sets for training, testing and validating
train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.125, random_state=42)

#creating copies to not write over data
train = train.copy()
val = val.copy()
test = test.copy()

#verifying data set lengths
#print(f"Training set: {len(train)}")
#print(f"Validation set: {len(val)}")
#print(f"Test set: {len(test)}")


#SEVERITY PREDICTION

#read in semantic lexicon
lexicon = pd.read_csv("HAZARD–SEMANTIC LEXICON.csv")

#creating a dictionary of lexicon keywords
keywords = lexicon['Keyword'].str.lower().tolist()
severity_dict = dict(zip(lexicon['Keyword'].str.lower(), lexicon['Severity']))

#looks inside each hazard description and finds which lexicon keywords appear inside it
def find_keywords(description):
    if pd.isna(description):
        return []
    description_lower = description.lower()
    #if there is a match then it is added to list
    matched = [kw for kw in keywords if kw in description_lower]
    return matched


#adds a new column to the data sets if keywords are found 
train['Matched Keywords'] = train['Hazard Description'].apply(find_keywords)
val['Matched Keywords'] = val['Hazard Description'].apply(find_keywords)
test['Matched Keywords'] = test['Hazard Description'].apply(find_keywords)

#if a lexicon match is found then the max severity between all values in the lexicon is returned
def compute_max_severity(matched_keywords):
    if not matched_keywords:
        return 0
    severities = [severity_dict[kw] for kw in matched_keywords if kw in severity_dict]
    return max(severities) if severities else 0

def compute_lexi_score(matched_keywords):
    if not matched_keywords:
        return 0.0
    severities = [severity_dict[kw] for kw in matched_keywords if kw in severity_dict]
    return float(np.mean(severities)) if severities else 0.0

#adds a new column to the data sets with this value 
train['LexiScore'] = train['Matched Keywords'].apply(compute_lexi_score)
val['LexiScore']   = val['Matched Keywords'].apply(compute_lexi_score)
test['LexiScore']  = test['Matched Keywords'].apply(compute_lexi_score)


#produces final severity using the original and lexicon value 
def final_severity(row):
    matched = row['Matched Keywords']
    if matched:
        severities = [severity_dict[kw] for kw in matched if kw in severity_dict]
        if severities:
            return max(severities)
    return row['Severity']

#adds a new column with this value
train['Final_Severity'] = train.apply(final_severity, axis=1)
val['Final_Severity'] = val.apply(final_severity, axis=1)
test['Final_Severity'] = test.apply(final_severity, axis=1)


#tf-idf implementation
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(train['Hazard Description'].fillna('')).toarray()
X_val_tfidf   = tfidf.transform(val['Hazard Description'].fillna('')).toarray()
X_test_tfidf  = tfidf.transform(test['Hazard Description'].fillna('')).toarray()

mlb = MultiLabelBinarizer()
X_train_keywords = mlb.fit_transform(train['Matched Keywords'])
X_val_keywords = mlb.transform(val['Matched Keywords'])
X_test_keywords = mlb.transform(test['Matched Keywords'])

#scaling 
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train[['LexiScore']])
X_val_num   = scaler.transform(val[['LexiScore']])
X_test_num  = scaler.transform(test[['LexiScore']])

X_train = np.hstack([
    X_train_tfidf,
    X_train_keywords,
    X_train_num
])

X_val = np.hstack([
    X_val_tfidf,
    X_val_keywords,
    X_val_num
])

X_test = np.hstack([
    X_test_tfidf,
    X_test_keywords,
    X_test_num
])

y_train = train['Final_Severity'].to_numpy()
y_val = val['Final_Severity'].to_numpy()
y_test = test['Final_Severity'].to_numpy() 

#MLP Classifier
model = MLPClassifier(
    hidden_layer_sizes=(128,64),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    alpha=0.0005,
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("=== MLP Classification Report ===")
print(classification_report(y_test, y_pred))

#KNN
knn_model = KNeighborsClassifier(
    n_neighbors=15,       
    weights='distance',  
    metric='euclidean'   
)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
print("=== KNN Classification Report ===")
print(classification_report(y_test, y_pred_knn))
print("KNN Accuracy:", metrics.accuracy_score(y_test, y_pred_knn))


#Support Vector Machine (SVM)
from sklearn.svm import LinearSVC

svm_model = LinearSVC(
    class_weight='balanced',
    max_iter=8000,
    random_state=42
)

svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("=== Linear SVM Report ===")
print(classification_report(y_test, y_pred_svm))
print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred_svm))

#DecisionTree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)

dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("=== Decision Tree Report ===")
print(classification_report(y_test, y_pred_dt))
print("DT Accuracy:", metrics.accuracy_score(y_test, y_pred_dt))


#Perceptron
from sklearn.linear_model import Perceptron

perc = Perceptron(
    penalty='l2',
    alpha=0.0001,
    max_iter=5000,
    class_weight='balanced',
    random_state=42
)

perc.fit(X_train, y_train)
y_pred_perc = perc.predict(X_test)

print("=== Perceptron Report ===")
print(classification_report(y_test, y_pred_perc))
print("Perceptron Accuracy:", metrics.accuracy_score(y_test, y_pred_perc))


#Confusion Matrices
fig, axs = plt.subplots(1, 5, figsize=(25, 5))

disp1 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axs[0])
axs[0].set_title("MLP Confusion Matrix")

disp2 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn, ax=axs[1])
axs[1].set_title("KNN Confusion Matrix")

disp3 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, ax=axs[2])
axs[2].set_title("LinearSVC Confusion Matrix")

disp4 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, ax=axs[3])
axs[3].set_title("Decision Tree Confusion Matrix")

disp5 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_perc, ax=axs[4])
axs[4].set_title("Perceptron Confusion Matrix")

plt.tight_layout()
plt.show()

#Accuracy charts
models = ["MLP", "KNN", "Linear SVM", "Decision Tree", "Perceptron"]
accuracies = [
    metrics.accuracy_score(y_test, y_pred),
    metrics.accuracy_score(y_test, y_pred_knn),
    metrics.accuracy_score(y_test, y_pred_svm),
    metrics.accuracy_score(y_test, y_pred_dt),
    metrics.accuracy_score(y_test, y_pred_perc)
]

plt.figure(figsize=(8,5))
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()



#without lexicon
y_train_nolex = train['Severity'].to_numpy()
y_val_nolex   = val['Severity'].to_numpy()
y_test_nolex  = test['Severity'].to_numpy()

mlp_nolex = MLPClassifier(
    hidden_layer_sizes=(128,64),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    alpha=0.0005,
    max_iter=1000,
    random_state=42
)

mlp_nolex.fit(X_train, y_train_nolex)
y_pred_mlp_nolex = mlp_nolex.predict(X_test)

print("=== MLP WITHOUT LEXICON ===")
print(classification_report(y_test_nolex, y_pred_mlp_nolex))
print("Without Lexicon Accuracy:", metrics.accuracy_score(y_test, y_pred_mlp_nolex))


fig, axs = plt.subplots(1, 2, figsize=(10, 4))

metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax=axs[0]
)
axs[0].set_title("MLP WITH Lexicon")

metrics.ConfusionMatrixDisplay.from_predictions(
    y_test_nolex, y_pred_mlp_nolex, ax=axs[1]
)
axs[1].set_title("MLP WITHOUT Lexicon")

plt.show()


models = ["MLP + Lexicon", "MLP Only"]
accuracies = [
    metrics.accuracy_score(y_test, y_pred),
    metrics.accuracy_score(y_test_nolex, y_pred_mlp_nolex)
]

plt.figure(figsize=(6,4))
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Impact of Hazard Semantic Lexicon")
plt.ylim(0, 1)
plt.show()

