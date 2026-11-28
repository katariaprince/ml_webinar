import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

st.title("Will this student pass")

@st.cache_resource
def load_model():
    students = pd.read_csv("students.csv")
    X = students[["hours_studied", "attendance", "prev_marks"]]
    y = students["passed"]
    return LogisticRegression(max_iter=1000).fit(X,y)

model = load_model()

hours = st.slider("Hours Studied", 1.0,10.0,5.0,step=0.1)
attendance = st.slider("Attendance %", 40,100,75)
prev_marks = st.slider("Previous Marks",0,100,60)

if st.button("Predict"):
    one = pd.DataFrame([[hours,attendance,prev_marks]],
                       columns=["hours_studied", "attendance", "prev_marks"])
    chance = model.predict_proba(one)[0][1]
    if chance>=0.5:
        st.success(f"PASS - {chance:.1%} confident")
    else:
        st.success(f"FAIL - {1-chance:.1%} confident")    






