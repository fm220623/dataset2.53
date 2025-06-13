import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide", page_title="Прогноз погоды", page_icon="🌤️")
st.markdown("""
### ℹ️ Описание переменных

- **Temperature** *(Температура)*: Температура в градусах Цельсия.
- **Humidity** *(Влажность)*: Процент влажности.
- **Wind Speed** *(Скорость ветра)*: Скорость ветра в км/ч.
- **Precipitation (%)** *(Осадки)*: Процент осадков.
- **Cloud Cover** *(Облачность)*: Категориальное описание облачного покрова.
- **Atmospheric Pressure** *(Атмосферное давление)*: Давление в гектопаскалях (hPa).
- **UV Index** *(УФ-индекс)*: Индекс ультрафиолетового излучения.
- **Season** *(Сезон)*: Сезон, во время которого были собраны данные.
- **Visibility (km)** *(Видимость)*: Видимость в километрах.
- **Location** *(Локация)*: Тип местности, где проводились измерения.
- **Weather Type** *(Тип погоды)*: Целевая переменная — указывает тип погоды для классификации.
""")

st.title("📚 Прогноз погоды с помощью ML")
st.write("Эта программа предсказывает тип погоды по параметрам, которые вы введете")

df = pd.read_csv("weather_clean.csv")
df = df.dropna()

st.subheader("Данные")
st.write(df.head())

st.subheader("График распределения типов погоды")
fig, ax = plt.subplots(figsize=(6, 3))
sns.countplot(x="Weather Type", data=df, ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)

# Готовим данные
y = df["Weather Type"]
X = df.drop("Weather Type", axis=1)
X = pd.get_dummies(X)

#MO
model = RandomForestClassifier()
model.fit(X, y)

st.sidebar.header("Введите параметры:")

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

user_data = pd.DataFrame([input_dict])
user_data = user_data.reindex(columns=X.columns, fill_value=0)

pred = model.predict(user_data)[0]
probs = model.predict_proba(user_data)[0]
labels = model.classes_

st.subheader("Введённые данные")
st.write(user_data)

st.subheader("Предсказание")
st.write("Тип погоды будет:", pred)

st.subheader("Вероятности")
st.write(pd.DataFrame([probs], columns=labels))
