import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

st.header('Category Prediction')

# Load and preprocess the data
bbc_text = pd.read_csv(r"C:\Users\HP\Documents\2 nd SEMESTER\NLP CLASS NOTES\bbc-text.txt")
bbc_text=bbc_text.rename(columns = {'text': 'News_Headline'}, inplace = False)
bbc_text.head()
bbc_text.category = bbc_text.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})
bbc_text.category.unique()
X = bbc_text.News_Headline
y = bbc_text.category
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)
vector = CountVectorizer(stop_words = 'english',lowercase=False)
# fit the vectorizer on the training data
vector.fit(X_train)
vector.vocabulary_
X_transformed = vector.transform(X_train)
X_transformed.toarray()
# for test data
X_test_transformed = vector.transform(X_test)
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)



# Get user input and make a prediction
input = st.text_area("Enter the text", value="")
if st.button("Predict"):
    vec = vectorizer.transform([input]).toarray()
    pred = naivebayes.predict(vec)[0]
    category = ['TECH', 'BUSINESS', 'SPORTS', 'ENTERTAINMENT', 'POLITICS'][pred]
    st.write("The predicted category is:", category)
