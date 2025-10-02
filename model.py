import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

DATA_PATH = "synthetic_dog_breed_health_data.csv"

# Настройка страницы
st.set_page_config(
    page_title="Предсказание здоровья собак",
    page_icon="🐕",
    layout="wide"
)

st.title("🐕 Предсказание здоровья собак")

# Функция обучения модели
def train_model(df):
    try:
        df = df.drop_duplicates()
        df = df.dropna(subset=['Healthy'])
        df = df.drop(columns=['ID', 'Synthetic'], errors='ignore')
        
        categorical_cols = [
            'Breed Size', 'Breed', 'Sex', 'Spay/Neuter Status',
            'Daily Activity Level', 'Diet', 'Food Brand',
            'Other Pets in Household', 'Medications', 'Seizures',
            'Owner Activity Level'
        ]
        
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
        
        df['Healthy'] = df['Healthy'].map({'Yes': 1, 'No': 0})
        
        df.fillna({
            'Age': df['Age'].median(),
            'Weight (lbs)': df['Weight (lbs)'].median(),
            'Daily Walk Distance (miles)': df['Daily Walk Distance (miles)'].median(),
            'Hours of Sleep': df['Hours of Sleep'].median(),
            'Play Time (hrs)': 0,
            'Annual Vet Visits': df['Annual Vet Visits'].median(),
            'Average Temperature (F)': df['Average Temperature (F)'].median()
        }, inplace=True)
        
        X = df.drop(columns=["Healthy"])
        y = df["Healthy"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, label_encoders, accuracy
        
    except Exception as e:
        st.error(f"Ошибка при обучении модели: {str(e)}")
        return None, None, None

# Функция генерации случайных данных собаки
def generate_dog_data(healthy=True):
    return {
        'Breed': random.choice([
            'Labrador Retriever', 'German Shepherd', 'Golden Retriever',
            'French Bulldog', 'Bulldog', 'Poodle', 'Beagle', 'Rottweiler',
            'Yorkshire Terrier', 'Dachshund', 'Siberian Husky', 'Boxer',
            'Border Collie', 'Cocker Spaniel', 'Australian Shepherd'
        ]),
        'Breed Size': random.choice(["Small", "Medium", "Large"]),
        'Sex': random.choice(["Male", "Female"]),
        'Age': random.uniform(1, 15) if healthy else random.uniform(8, 20),
        'Weight (lbs)': random.uniform(10, 80) if healthy else random.uniform(5, 150),
        'Spay/Neuter Status': random.choice(["Spayed/Neutered", "Intact"]),
        'Daily Activity Level': random.choice(["Medium", "High"]) if healthy else random.choice(["Low", "Medium"]),
        'Diet': random.choice(["Dry Food", "Wet Food", "Raw Diet", "Mixed"]),
        'Food Brand': random.choice(["Premium", "Standard"]) if healthy else random.choice(["Budget", "Standard"]),
        'Daily Walk Distance (miles)': random.uniform(1, 5) if healthy else random.uniform(0, 2),
        'Other Pets in Household': random.choice(["Yes", "No"]),
        'Medications': "No" if healthy else "Yes",
        'Seizures': "No" if healthy else random.choice(["Yes", "No"]),
        'Hours of Sleep': random.uniform(10, 14) if healthy else random.uniform(4, 18),
        'Play Time (hrs)': random.uniform(1, 4) if healthy else random.uniform(0, 1),
        'Owner Activity Level': random.choice(["Medium", "High"]) if healthy else random.choice(["Low", "Medium"]),
        'Annual Vet Visits': random.randint(1, 3) if healthy else random.randint(3, 8),
        'Average Temperature (F)': random.uniform(60, 75) if healthy else random.uniform(20, 100),
    }

# Функция для создания формы ручного ввода данных
def create_manual_input_form():
    # Получаем значения из session_state если они есть, иначе используем значения по умолчанию
    default_values = st.session_state.get("form_values", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        breed = st.selectbox(
            "Порода",
            ['Labrador Retriever', 'German Shepherd', 'Golden Retriever',
             'French Bulldog', 'Bulldog', 'Poodle', 'Beagle', 'Rottweiler',
             'Yorkshire Terrier', 'Dachshund', 'Siberian Husky', 'Boxer',
             'Border Collie', 'Cocker Spaniel', 'Australian Shepherd'],
            index=['Labrador Retriever', 'German Shepherd', 'Golden Retriever',
                   'French Bulldog', 'Bulldog', 'Poodle', 'Beagle', 'Rottweiler',
                   'Yorkshire Terrier', 'Dachshund', 'Siberian Husky', 'Boxer',
                   'Border Collie', 'Cocker Spaniel', 'Australian Shepherd'].index(
                       default_values.get('Breed', 'Labrador Retriever'))
        )
        
        breed_size = st.selectbox(
            "Размер породы", 
            ["Small", "Medium", "Large"],
            index=["Small", "Medium", "Large"].index(default_values.get('Breed Size', 'Medium'))
        )
        
        sex = st.selectbox(
            "Пол", 
            ["Male", "Female"],
            index=["Male", "Female"].index(default_values.get('Sex', 'Male'))
        )
        
        age = st.slider(
            "Возраст (лет)", 
            0.5, 20.0, 
            float(default_values.get('Age', 5.0)), 
            0.5
        )
        
        weight = st.slider(
            "Вес (фунты)", 
            5.0, 150.0, 
            float(default_values.get('Weight (lbs)', 50.0)), 
            1.0
        )
        
        spay_neuter = st.selectbox(
            "Статус стерилизации", 
            ["Spayed/Neutered", "Intact"],
            index=["Spayed/Neutered", "Intact"].index(default_values.get('Spay/Neuter Status', 'Spayed/Neutered'))
        )
    
    with col2:
        activity_level = st.selectbox(
            "Уровень активности", 
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(default_values.get('Daily Activity Level', 'Medium'))
        )
        
        diet = st.selectbox(
            "Диета", 
            ["Dry Food", "Wet Food", "Raw Diet", "Mixed"],
            index=["Dry Food", "Wet Food", "Raw Diet", "Mixed"].index(default_values.get('Diet', 'Dry Food'))
        )
        
        food_brand = st.selectbox(
            "Качество корма", 
            ["Budget", "Standard", "Premium"],
            index=["Budget", "Standard", "Premium"].index(default_values.get('Food Brand', 'Standard'))
        )
        
        walk_distance = st.slider(
            "Дистанция прогулок (мили в день)", 
            0.0, 10.0, 
            float(default_values.get('Daily Walk Distance (miles)', 2.0)), 
            0.1
        )
        
        other_pets = st.selectbox(
            "Другие питомцы в доме", 
            ["Yes", "No"],
            index=["Yes", "No"].index(default_values.get('Other Pets in Household', 'No'))
        )
        
        medications = st.selectbox(
            "Прием лекарств", 
            ["Yes", "No"],
            index=["Yes", "No"].index(default_values.get('Medications', 'No'))
        )
    
    with col3:
        seizures = st.selectbox(
            "Судороги", 
            ["Yes", "No"],
            index=["Yes", "No"].index(default_values.get('Seizures', 'No'))
        )
        
        sleep_hours = st.slider(
            "Часы сна в день", 
            4.0, 18.0, 
            float(default_values.get('Hours of Sleep', 12.0)), 
            0.5
        )
        
        play_time = st.slider(
            "Время игр (часы в день)", 
            0.0, 8.0, 
            float(default_values.get('Play Time (hrs)', 2.0)), 
            0.1
        )
        
        owner_activity = st.selectbox(
            "Активность владельца", 
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(default_values.get('Owner Activity Level', 'Medium'))
        )
        
        vet_visits = st.slider(
            "Визиты к ветеринару в год", 
            0, 10, 
            int(default_values.get('Annual Vet Visits', 2)), 
            1
        )
        
        avg_temp = st.slider(
            "Средняя температура (°F)", 
            20.0, 100.0, 
            float(default_values.get('Average Temperature (F)', 70.0)), 
            1.0
        )
    
    # Три кнопки для действий
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔍 Предсказать здоровье", type="primary", use_container_width=True):
            manual_data = {
                'Breed': breed,
                'Breed Size': breed_size,
                'Sex': sex,
                'Age': age,
                'Weight (lbs)': weight,
                'Spay/Neuter Status': spay_neuter,
                'Daily Activity Level': activity_level,
                'Diet': diet,
                'Food Brand': food_brand,
                'Daily Walk Distance (miles)': walk_distance,
                'Other Pets in Household': other_pets,
                'Medications': medications,
                'Seizures': seizures,
                'Hours of Sleep': sleep_hours,
                'Play Time (hrs)': play_time,
                'Owner Activity Level': owner_activity,
                'Annual Vet Visits': vet_visits,
                'Average Temperature (F)': avg_temp,
            }
            st.session_state["generated_dog"] = manual_data
            st.session_state["input_method"] = "manual"
            st.session_state["form_values"] = manual_data
            st.rerun()
    
    with col_btn2:
        if st.button("🎲 Здоровая собака", use_container_width=True):
            healthy_data = generate_dog_data(healthy=True)
            st.session_state["form_values"] = healthy_data
            st.session_state["generated_dog"] = healthy_data
            st.session_state["input_method"] = "generated_healthy"
            st.rerun()
    
    with col_btn3:
        if st.button("🎲 Нездоровая собака", use_container_width=True):
            unhealthy_data = generate_dog_data(healthy=False)
            st.session_state["form_values"] = unhealthy_data
            st.session_state["generated_dog"] = unhealthy_data
            st.session_state["input_method"] = "generated_unhealthy"
            st.rerun()

# Функция отображения результатов предсказания
def show_prediction_results(input_data, model, label_encoders, input_method="generated"):        
    encoded_data = input_data.copy()
    for col, le in label_encoders.items():
        if col in encoded_data:
            try:
                encoded_data[col] = le.transform([encoded_data[col]])[0]
            except ValueError:
                encoded_data[col] = 0
    
    df_pred = pd.DataFrame([encoded_data])
    prediction = model.predict(df_pred)[0]
    probabilities = model.predict_proba(df_pred)[0]
    
    st.subheader("📊 Результат предсказания")
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        status_color = "🟢" if prediction == 1 else "🔴"
        st.metric("Статус", f"{status_color} {'Здоровая' if prediction == 1 else 'Нездоровая'}")
    with col_res2:
        st.metric("Вероятность здоровья", f"{probabilities[1]*100:.1f}%")
    with col_res3:
        st.metric("Риск проблем", f"{probabilities[0]*100:.1f}%")
    
uploaded_file = pd.read_csv(DATA_PATH)

if uploaded_file is not None:
    df = pd.read_csv(DATA_PATH)
    
    with st.spinner("🔄 Обучаю модель..."):
        model, label_encoders, accuracy = train_model(df)
    
    if model is not None:
        st.success(f"🎯 Модель успешно обучена! Точность: {accuracy:.1%}")
        
        # Объединенная форма ввода данных
        st.header("🎯 Введите данные собаки")
        create_manual_input_form()
        
        # Отображение результатов
        if "generated_dog" in st.session_state:
            input_data = st.session_state["generated_dog"]
            method = st.session_state.get("input_method", "generated")
            show_prediction_results(input_data, model, label_encoders, method)
            

