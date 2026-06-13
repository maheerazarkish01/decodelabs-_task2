import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.title("🚀 Asteroid Risk Prediction Dashboard")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # ---------------- FIND COLUMNS ----------------
    distance_col = None
    velocity_col = None

    for col in df.columns:
        if "distance" in col.lower():
            distance_col = col
        if "velocity" in col.lower() or "speed" in col.lower():
            velocity_col = col

    st.write("✅ Distance Column:", distance_col)
    st.write("✅ Velocity Column:", velocity_col)

    # ---------------- CHECK ----------------
    if distance_col is None or velocity_col is None:
        st.error("❌ Could not find required columns. Check dataset!")
    else:

        # ---------------- CREATE RISK ----------------
        def risk_label(row):
            if row[distance_col] < 5000000 and row[velocity_col] > 20000:
                return "Dangerous"
            else:
                return "Safe"

        df['Risk'] = df.apply(risk_label, axis=1)

        st.subheader("✅ Data with Risk Column")
        st.write(df.head())

        # ---------------- VISUALIZATION ----------------
        st.subheader("📈 Scatter Plot")
        fig1, ax1 = plt.subplots()
        ax1.scatter(df[distance_col], df[velocity_col])
        ax1.set_xlabel(distance_col)
        ax1.set_ylabel(velocity_col)
        st.pyplot(fig1)

        st.subheader("📊 Distance Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df[distance_col], bins=20)
        ax2.set_xlabel(distance_col)
        st.pyplot(fig2)

        # ---------------- MODEL ----------------
        X = df[[distance_col, velocity_col]]
        y = df['Risk']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)

        st.subheader("🤖 Model Accuracy")
        st.write(f"Accuracy: {acc:.2f}")

        # ---------------- USER INPUT ----------------
        st.subheader("🔮 Predict Asteroid Risk")

        user_distance = st.number_input("Enter Distance")
        user_velocity = st.number_input("Enter Velocity")

        if st.button("Predict"):
            prediction = model.predict([[user_distance, user_velocity]])
            st.success(f"Prediction: {prediction[0]}")