import pickle
saved_model = pickle.dumps(naivebayes)
s = pickle.loads(saved_model)


# Get user input and make a prediction
input = st.text_area("Enter the text", value="")
if st.button("Predict"):
    vec = vector.transform(input).toarray()
    pred = s.predict(vec)[0]
    category = ['TECH', 'BUSINESS', 'SPORTS', 'ENTERTAINMENT', 'POLITICS'][pred]
    st.write("The predicted category is:", category)
