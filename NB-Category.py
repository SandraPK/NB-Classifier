import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

st.header('Category Prediction')

# Load and preprocess the data
with open('bbc-text.txt', 'r') as file:
    data = file.read()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['category'], test_size=0.2, random_state=42)

# Convert the text data into a bag-of-words representation
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier on the training data
naivebayes = MultinomialNB()
naivebayes.fit(X_train_vec, y_train)

# Get user input and make a prediction
input = st.text_area("Enter the text", value="")
if st.button("Predict"):
    vec = vectorizer.transform([input]).toarray()
    pred = naivebayes.predict(vec)[0]
    category = ['TECH', 'BUSINESS', 'SPORTS', 'ENTERTAINMENT', 'POLITICS'][pred]
    st.write("The predicted category is:", category)
