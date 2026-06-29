


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Categories to classify
categories = [
    'alt.atheism',
    'soc.religion.christian',
    'comp.graphics',
    'sci.med'
]

# Load dataset
newsgroups = fetch_20newsgroups(
    subset='all',
    categories=categories,
    shuffle=True,
    random_state=42
)

# Train-test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    newsgroups.data,
    newsgroups.target,
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Validation performance
y_val_pred = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print("Validation Accuracy:", val_accuracy)
print("\nClassification Report (Validation Set):")
print(classification_report(
    y_val,
    y_val_pred,
    target_names=newsgroups.target_names
))

# Test performance
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nTest Accuracy:", test_accuracy)
print("\nClassification Report (Test Set):")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=newsgroups.target_names
))
