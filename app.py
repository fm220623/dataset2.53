import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide", page_title="Прогноз погоды", page_icon="🌤️")

st.title("📚 Прогноз погоды с помощью ML")
st.write("Эта программа предсказывает тип погоды по параметрам, которые вы введете")

# Загружаем данные
df = pd.read_csv("weather_classification_data.csv")
df = df.dropna()

st.subheader("Данные")
st.write(df.head())

# График (небольшой)
st.subheader("График распределения типов погоды")
fig, ax = plt.subplots(figsize=(6, 3))
sns.countplot(x="Weather Type", data=df, ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)

# Готовим данные
y = df["Weather Type"]
X = df.drop("Weather Type", axis=1)
X = pd.get_dummies(X)

# Обучаем модель
model = RandomForestClassifier()
model.fit(X, y)

st.sidebar.header("Введите параметры:")

# Ввод пользователя
input_dict = {}
for col in df.columns:
    if col == "Weather Type":
        continue
    if df[col].dtype == "object":
        val = st.sidebar.selectbox(col, df[col].unique())
        for dcol in pd.get_dummies(df[[col]]).columns:
            input_dict[dcol] = 1 if dcol == f"{col}_{val}" else 0
    else:
        val = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_dict[col] = val

# Подгоняем к нужному формату
user_data = pd.DataFrame([input_dict])
user_data = user_data.reindex(columns=X.columns, fill_value=0)

# Предсказание
pred = model.predict(user_data)[0]
probs = model.predict_proba(user_data)[0]
labels = model.classes_

st.subheader("Введённые данные")
st.write(user_data)

st.subheader("Предсказание")
st.write("Тип погоды будет:", pred)

st.subheader("Вероятности")
st.write(pd.DataFrame([probs], columns=labels))
