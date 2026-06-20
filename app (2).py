
import streamlit as st
import joblib
import numpy as np

model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details:")

age = st.number_input("Age", 0, 100, 25)

sex = st.selectbox("Sex", ["male", "female"])
sex = 1 if sex == "male" else 0

pclass = st.selectbox("Pclass", [1, 2, 3])

fare = st.number_input("Fare", 0.0, 500.0, 50.0)

embarked = st.selectbox("Embarked", ["S", "C", "Q"])
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

if st.button("Predict"):

    input_data = np.array([[pclass, sex, age, fare, embarked, family_size, is_alone]])

    proba = model.predict_proba(input_data)[0]

    st.subheader("📊 Result")
    st.write(f"❌ Not Survived: {proba[0]:.2f}")
    st.write(f"🎉 Survived: {proba[1]:.2f}")

    st.progress(float(proba[1]))
