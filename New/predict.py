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

# Accuracy score
acc_score = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", acc_score)

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

fig, axs = plt.subplots(1,2, figsize=(12,5))

disp1 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axs[0])
axs[0].set_title("MLP Confusion Matrix")

disp2 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn, ax=axs[1])
axs[1].set_title("KNN Confusion Matrix")

plt.tight_layout()
plt.show()

#print(model.predict("Slips and trips"))


#LIKELIHOOD PREDICTION
# LIKELIHOOD PREDICTION
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

train['text'] = train['Hazard Name'].fillna('') + ' ' + train['Hazard Description'].fillna('')
val['text'] = val['Hazard Name'].fillna('') + ' ' + val['Hazard Description'].fillna('')
test['text'] = test['Hazard Name'].fillna('') + ' ' + test['Hazard Description'].fillna('')


#load likelihood lexicon
likelihood_lexicon = pd.read_csv("HAZARD–SEMANTIC LEXICON likelihood.csv")

# CLEAN keywords safely
likelihood_lexicon['Keyword'] = (
    likelihood_lexicon['Keyword']
    .astype(str)
    .str.lower()
    .str.strip()
)

# Remove invalid rows
likelihood_lexicon = likelihood_lexicon[
    likelihood_lexicon['Keyword'].notna() &
    (likelihood_lexicon['Keyword'] != '') &
    (likelihood_lexicon['Keyword'] != 'nan')
]

likelihood_keywords = likelihood_lexicon['Keyword'].tolist()

likelihood_dict = dict(zip(
    likelihood_lexicon['Keyword'],
    likelihood_lexicon['Likelihood']
))


def find_likelihood_keywords(text):
    if pd.isna(text):
        return []
    text = text.lower()
    return [kw for kw in likelihood_keywords if kw in text]


train['Matched_Likelihood_Keywords'] = train['text'].apply(find_likelihood_keywords)
val['Matched_Likelihood_Keywords']   = val['text'].apply(find_likelihood_keywords)
test['Matched_Likelihood_Keywords']  = test['text'].apply(find_likelihood_keywords)


def compute_max_likelihood(matched):
    if not matched:
        return 0
    values = [likelihood_dict[k] for k in matched if k in likelihood_dict]
    return max(values) if values else 0

def compute_mean_likelihood(matched):
    if not matched:
        return 0.0
    values = [likelihood_dict[k] for k in matched if k in likelihood_dict]
    return float(np.mean(values)) if values else 0.0


for df_split in (train, val, test):
    df_split['Lexicon_Max_Likelihood'] = df_split['Matched_Likelihood_Keywords'].apply(compute_max_likelihood)
    df_split['Lexicon_Mean_Likelihood'] = df_split['Matched_Likelihood_Keywords'].apply(compute_mean_likelihood)

scaler_likelihood = StandardScaler()

train_lik_num = scaler_likelihood.fit_transform(
    train[['Lexicon_Max_Likelihood', 'Lexicon_Mean_Likelihood']]
)
val_lik_num = scaler_likelihood.transform(
    val[['Lexicon_Max_Likelihood', 'Lexicon_Mean_Likelihood']]
)
test_lik_num = scaler_likelihood.transform(
    test[['Lexicon_Max_Likelihood', 'Lexicon_Mean_Likelihood']]
)


#TF-IDF for likelihood
tfidf_likelihood = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),   
    max_features=5000
)

X_train_likelihood = np.hstack([
    tfidf_likelihood.fit_transform(train['text']).toarray(),
    train_lik_num
])

X_val_likelihood = np.hstack([
    tfidf_likelihood.transform(val['text']).toarray(),
    val_lik_num
])

X_test_likelihood = np.hstack([
    tfidf_likelihood.transform(test['text']).toarray(),
    test_lik_num
])


y_train_likelihood = train['Likelihood'].to_numpy()
y_val_likelihood   = val['Likelihood'].to_numpy()
y_test_likelihood  = test['Likelihood'].to_numpy()


# train MLP with weights
model_likelihood = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    alpha=0.0005,
    max_iter=1000,
    random_state=42
)

# Pass sample_weight based on class weight
model_likelihood.fit(X_train_likelihood, y_train_likelihood)


y_pred_likelihood = model_likelihood.predict(X_test_likelihood)

print("=== Likelihood Classification Report ===")
print(classification_report(y_test_likelihood, y_pred_likelihood, digits=3))

acc_likelihood = accuracy_score(y_test_likelihood, y_pred_likelihood)
print("Likelihood Accuracy:", acc_likelihood)

disp = ConfusionMatrixDisplay.from_predictions(y_test_likelihood, y_pred_likelihood)
disp.figure_.suptitle("Likelihood Confusion Matrix")
plt.tight_layout()
plt.show()

# Severity
joblib.dump(model, "mlp_severity_model.pkl")
joblib.dump(tfidf, "tfidf_severity.pkl")
joblib.dump(mlb, "mlb_severity.pkl")
joblib.dump(scaler, "scaler_severity.pkl")

# Likelihood
joblib.dump(model_likelihood, "mlp_likelihood_model.pkl")
joblib.dump(tfidf_likelihood, "tfidf_likelihood.pkl")
joblib.dump(scaler_likelihood, "scaler_likelihood.pkl")