# import streamlit as st
# import os
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import cv2
# import time
# import json

# # 1. Streamlit Page Config
# st.set_page_config(page_title="Indian Food Nutrition Analyzer", layout="wide")

# # 2. Load Custom Model and Labels
# @st.cache_resource
# def load_model():
#     try:
#         model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indian_food_model.h5")
#         model = tf.keras.models.load_model(model_path)
#         class_labels = sorted([
#             "aloo_sabji", "baingan_bharta", "besan_chilla", "bhel_puri", "bhindi_masala",
#             "chana_masala", "chole_bhature", "dosa", "egg", "gajar_ka_halwa", "idli", "jalebi",
#             "kadai_paneer", "khichdi", "laddu", "medu_vada", "momos", "paneer_butter_masala",
#             "pani_puri", "paratha", "pav_bhaji", "plain_rice", "poha", "puri_bhaji", "rajma_chawal",
#             "roti", "samosa", "upma", "vada_pav"
#         ])
#         return model, class_labels
#     except Exception as e:
#         st.error(f"Model loading failed: {str(e)}")
#         return None, []

# model, class_labels = load_model()

# # 3. Local Indian Food Nutrition Database
# indian_nutrition_db = {
#     "aloo sabji": {"grams": 100, "calories": 150, "protein": 3, "carbs": 18, "fat": 8, "sugar": 2, "fiber": 2},
#     "baingan bharta": {"grams": 100, "calories": 120, "protein": 2, "carbs": 14, "fat": 7, "sugar": 5, "fiber": 4},
#     "besan chilla": {"grams": 100, "calories": 180, "protein": 9, "carbs": 15, "fat": 10, "sugar": 1, "fiber": 3},
#     "bhel puri": {"grams": 100, "calories": 250, "protein": 6, "carbs": 32, "fat": 12, "sugar": 4, "fiber": 3},
#     "bhindi masala": {"grams": 100, "calories": 100, "protein": 2, "carbs": 10, "fat": 6, "sugar": 3, "fiber": 3},
#     "chana masala": {"grams": 100, "calories": 190, "protein": 10, "carbs": 28, "fat": 5, "sugar": 6, "fiber": 6},
#     "chole bhature": {"grams": 100, "calories": 400, "protein": 12, "carbs": 35, "fat": 22, "sugar": 3, "fiber": 4},
#     "dosa": {"grams": 100, "calories": 210, "protein": 5, "carbs": 30, "fat": 10, "sugar": 1, "fiber": 2},
#     "egg": {"grams": 100, "calories": 70, "protein": 6, "carbs": 1, "fat": 5, "sugar": 0, "fiber": 0},
#     "gajar ka halwa": {"grams": 100, "calories": 320, "protein": 4.5, "carbs": 40, "fat": 16, "sugar": 28, "fiber": 2},
#     "idli": {"grams": 1, "calories": 60, "protein": 2, "carbs": 12, "fat": 0.4, "sugar": 0.3, "fiber": 0.5},
#     "jalebi": {"grams": 100, "calories": 350, "protein": 2, "carbs": 45, "fat": 18, "sugar": 35, "fiber": 1},
#     "kadai paneer": {"grams": 100, "calories": 300, "protein": 11, "carbs": 12, "fat": 24, "sugar": 4, "fiber": 2},
#     "khichdi": {"grams": 100, "calories": 180, "protein": 6, "carbs": 28, "fat": 5, "sugar": 2, "fiber": 2},
#     "laddu": {"grams": 100, "calories": 200, "protein": 3, "carbs": 22, "fat": 10, "sugar": 16, "fiber": 1},
#     "medu vada": {"grams": 100, "calories": 150, "protein": 4, "carbs": 20, "fat": 7, "sugar": 1, "fiber": 2},
#     "momos": {"grams": 100, "calories": 140, "protein": 6, "carbs": 20, "fat": 4, "sugar": 1, "fiber": 1},
#     "paneer butter masala": {"grams": 100, "calories": 400, "protein": 12, "carbs": 14, "fat": 35, "sugar": 5, "fiber": 1},
#     "pani puri": {"grams": 100, "calories": 220, "protein": 4, "carbs": 30, "fat": 10, "sugar": 2, "fiber": 1},
#     "paratha": {"grams": 100, "calories": 280, "protein": 6, "carbs": 32, "fat": 12, "sugar": 1, "fiber": 2},
#     "pav bhaji": {"grams": 100, "calories": 400, "protein": 7, "carbs": 45, "fat": 18, "sugar": 6, "fiber": 4},
#     "plain rice": {"grams": 100, "calories": 130, "protein": 2.5, "carbs": 28, "fat": 0.3, "sugar": 0.2, "fiber": 0.5},
#     "poha": {"grams": 100, "calories": 180, "protein": 4, "carbs": 26, "fat": 8, "sugar": 2, "fiber": 1.5},
#     "puri bhaji": {"grams": 100, "calories": 320, "protein": 5, "carbs": 30, "fat": 18, "sugar": 3, "fiber": 2},
#     "rajma chawal": {"grams": 100, "calories": 300, "protein": 10, "carbs": 40, "fat": 8, "sugar": 3, "fiber": 6},
#     "roti": {"grams": 100, "calories": 100, "protein": 3, "carbs": 20, "fat": 1, "sugar": 0.5, "fiber": 2},
#     "samosa": {"grams": 100, "calories": 280, "protein": 5, "carbs": 32, "fat": 14, "sugar": 2, "fiber": 2},
#     "upma": {"grams": 100, "calories": 160, "protein": 4, "carbs": 22, "fat": 6, "sugar": 1, "fiber": 1},
#     "vada pav": {"grams": 1, "calories": 300, "protein": 5, "carbs": 35, "fat": 14, "sugar": 2, "fiber": 2}
# }

# # 4. Prediction Function
# def predict_food(img, model, class_labels):
#     if model is None or img is None:
#         return "model_not_loaded", 0.0

#     try:
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((224, 224))
#         img_array = np.array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         predictions = model.predict(img_array)[0]
#         class_idx = np.argmax(predictions)
#         confidence = float(predictions[class_idx])
#         predicted_label = class_labels[class_idx]
#         return predicted_label.replace("_", " "), confidence
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
#         return "prediction_error", 0.0

# # 5. Upload & Predict Section
# st.title("🍽️ Indian Food Nutrition Analyzer")
# st.markdown("Upload an Indian food image and we'll analyze its nutritional content!")

# uploaded = st.file_uploader("Upload Food Image", type=['jpg', 'jpeg', 'png'])

# if uploaded:
#     file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     if img is not None:
#         st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

#         input_mode = st.radio("Select input mode", ["Grams", "Number of pieces"], index=0)

#         quantity = st.number_input(
#             "Enter Quantity", min_value=1, value=100 if input_mode == "Grams" else 1, step=1
#         )

#         if st.button("🤖 Analyze Nutrition"):
#             with st.spinner("Analyzing..."):
#                 food_class, confidence = predict_food(img, model, class_labels)

#             if food_class not in ["model_not_loaded", "prediction_error"]:
#                 st.markdown(f"### 🍽️ Predicted: `{food_class}` ({confidence * 100:.2f}% confidence)")
#                 st.markdown(f"### 📊 Quantity: `{quantity} {'grams' if input_mode == 'Grams' else 'piece(s)'}`")

#                 food_key = food_class.lower().strip()
#                 if food_key in indian_nutrition_db:
#                     base = indian_nutrition_db[food_key]
#                     unit = base['grams']
#                     factor = (quantity / 100) if input_mode == "Grams" else quantity
#                     if input_mode == "Grams" and unit != 100:
#                         factor = quantity / unit
#                     elif input_mode == "Number of pieces" and unit == 100:
#                         factor = (quantity * 100) / 100

#                     nutrition = {k: round(v * factor, 2) for k, v in base.items() if k != 'grams'}

#                     st.success("Nutrition Data (from Local Indian Database)")
#                     cols = st.columns(3)
#                     cols[0].metric("Calories", f"{nutrition['calories']} kcal")
#                     cols[1].metric("Protein", f"{nutrition['protein']} g")
#                     cols[2].metric("Carbs", f"{nutrition['carbs']} g")

#                     cols = st.columns(3)
#                     cols[0].metric("Fat", f"{nutrition['fat']} g")
#                     cols[1].metric("Sugar", f"{nutrition['sugar']} g")
#                     cols[2].metric("Fiber", f"{nutrition['fiber']} g")
#                 else:
#                     st.warning(f"'{food_class}' not found in local database. Please consider adding it manually.")
#             else:
#                 st.error("Food could not be predicted. Try another image.")
import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time
import json
import datetime
from utils import firebase_config
from firebase_admin import auth, firestore

# 1. Streamlit Page Config
st.set_page_config(page_title="Indian Food Nutrition Analyzer", layout="wide")

# Session State for User
if 'user' not in st.session_state:
    st.session_state.user = None

# 2. Firebase Auth Section
def auth_section():
    st.sidebar.header("🔐 User Login")

    if st.session_state.user:
        st.sidebar.success(f"Logged in as: {st.session_state.user['email']}")
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()
        return True

    choice = st.sidebar.radio("Choose:", ["Login", "Sign Up"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if choice == "Sign Up":
        if st.sidebar.button("Create Account"):
            try:
                user = auth.create_user(email=email, password=password)
                st.success("✅ Account created! You can now log in.")
            except Exception as e:
                st.error(f"Error: {e}")

    if choice == "Login":
        if st.sidebar.button("Login"):
            try:
                user = auth.get_user_by_email(email)
                st.session_state.user = {
                    'email': email,
                    'uid': user.uid
                }
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    return False

# 3. Load Custom Model and Labels
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indian_food_model.h5")
        model = tf.keras.models.load_model(model_path)
        class_labels = sorted([
            "aloo_sabji", "baingan_bharta", "besan_chilla", "bhel_puri", "bhindi_masala",
            "chana_masala", "chole_bhature", "dosa", "egg", "gajar_ka_halwa", "idli", "jalebi",
            "kadai_paneer", "khichdi", "laddu", "medu_vada", "momos", "paneer_butter_masala",
            "pani_puri", "paratha", "pav_bhaji", "plain_rice", "poha", "puri_bhaji", "rajma_chawal",
            "roti", "samosa", "upma", "vada_pav"
        ])
        return model, class_labels
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, []

model, class_labels = load_model()

indian_nutrition_db = {
    "aloo sabji": {"grams": 100, "calories": 150, "protein": 3, "carbs": 18, "fat": 8, "sugar": 2, "fiber": 2},
    "baingan bharta": {"grams": 100, "calories": 120, "protein": 2, "carbs": 14, "fat": 7, "sugar": 5, "fiber": 4},
    "besan chilla": {"grams": 100, "calories": 180, "protein": 9, "carbs": 15, "fat": 10, "sugar": 1, "fiber": 3},
    "bhel puri": {"grams": 100, "calories": 250, "protein": 6, "carbs": 32, "fat": 12, "sugar": 4, "fiber": 3},
    "bhindi masala": {"grams": 100, "calories": 100, "protein": 2, "carbs": 10, "fat": 6, "sugar": 3, "fiber": 3},
    "chana masala": {"grams": 100, "calories": 190, "protein": 10, "carbs": 28, "fat": 5, "sugar": 6, "fiber": 6},
    "chole bhature": {"grams": 100, "calories": 400, "protein": 12, "carbs": 35, "fat": 22, "sugar": 3, "fiber": 4},
    "dosa": {"grams": 100, "calories": 210, "protein": 5, "carbs": 30, "fat": 10, "sugar": 1, "fiber": 2},
    "egg": {"grams": 100, "calories": 70, "protein": 6, "carbs": 1, "fat": 5, "sugar": 0, "fiber": 0},
    "gajar ka halwa": {"grams": 100, "calories": 320, "protein": 4.5, "carbs": 40, "fat": 16, "sugar": 28, "fiber": 2},
    "idli": {"grams": 1, "calories": 60, "protein": 2, "carbs": 12, "fat": 0.4, "sugar": 0.3, "fiber": 0.5},
    "jalebi": {"grams": 100, "calories": 350, "protein": 2, "carbs": 45, "fat": 18, "sugar": 35, "fiber": 1},
    "kadai paneer": {"grams": 100, "calories": 300, "protein": 11, "carbs": 12, "fat": 24, "sugar": 4, "fiber": 2},
    "khichdi": {"grams": 100, "calories": 180, "protein": 6, "carbs": 28, "fat": 5, "sugar": 2, "fiber": 2},
    "laddu": {"grams": 100, "calories": 200, "protein": 3, "carbs": 22, "fat": 10, "sugar": 16, "fiber": 1},
    "medu vada": {"grams": 100, "calories": 150, "protein": 4, "carbs": 20, "fat": 7, "sugar": 1, "fiber": 2},
    "momos": {"grams": 100, "calories": 140, "protein": 6, "carbs": 20, "fat": 4, "sugar": 1, "fiber": 1},
    "paneer butter masala": {"grams": 100, "calories": 400, "protein": 12, "carbs": 14, "fat": 35, "sugar": 5, "fiber": 1},
    "pani puri": {"grams": 100, "calories": 220, "protein": 4, "carbs": 30, "fat": 10, "sugar": 2, "fiber": 1},
    "paratha": {"grams": 100, "calories": 280, "protein": 6, "carbs": 32, "fat": 12, "sugar": 1, "fiber": 2},
    "pav bhaji": {"grams": 100, "calories": 400, "protein": 7, "carbs": 45, "fat": 18, "sugar": 6, "fiber": 4},
    "plain rice": {"grams": 100, "calories": 130, "protein": 2.5, "carbs": 28, "fat": 0.3, "sugar": 0.2, "fiber": 0.5},
    "poha": {"grams": 100, "calories": 180, "protein": 4, "carbs": 26, "fat": 8, "sugar": 2, "fiber": 1.5},
    "puri bhaji": {"grams": 100, "calories": 320, "protein": 5, "carbs": 30, "fat": 18, "sugar": 3, "fiber": 2},
    "rajma chawal": {"grams": 100, "calories": 300, "protein": 10, "carbs": 40, "fat": 8, "sugar": 3, "fiber": 6},
    "roti": {"grams": 100, "calories": 100, "protein": 3, "carbs": 20, "fat": 1, "sugar": 0.5, "fiber": 2},
    "samosa": {"grams": 100, "calories": 280, "protein": 5, "carbs": 32, "fat": 14, "sugar": 2, "fiber": 2},
    "upma": {"grams": 100, "calories": 160, "protein": 4, "carbs": 22, "fat": 6, "sugar": 1, "fiber": 1},
    "vada pav": {"grams": 1, "calories": 300, "protein": 5, "carbs": 35, "fat": 14, "sugar": 2, "fiber": 2}
}

def predict_food(img, model, class_labels):
    if model is None or img is None:
        return "model_not_loaded", 0.0

    try:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx])
        predicted_label = class_labels[class_idx]
        return predicted_label.replace("_", " "), confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "prediction_error", 0.0

# 6. Log Meal to Firestore
def log_meal_to_firestore(food, nutrition, quantity, input_mode):
    if not st.session_state.user:
        return
    meal_data = {
        'food': food,
        'quantity': quantity,
        'input_mode': input_mode,
        'nutrition': nutrition,
        'timestamp': datetime.datetime.now()
    }
    doc_ref = firebase_config.db.collection("users")\
        .document(st.session_state.user['uid'])\
        .collection("meals").document()
    doc_ref.set(meal_data)

# 7. Show Meal History
def show_meal_history():
    if not st.session_state.user:
        return

    st.subheader("📅 Meal History")
    ref = firebase_config.db.collection("users")\
        .document(st.session_state.user['uid'])\
        .collection("meals")\
        .order_by("timestamp", direction=firestore.Query.DESCENDING)\
        .limit(10).stream()

    for doc in ref:
        meal = doc.to_dict()
        st.markdown(f"### 🥘 {meal['food'].title()} ({meal['quantity']} {meal['input_mode']})")
        cols = st.columns(3)
        cols[0].metric("Calories", f"{meal['nutrition']['calories']} kcal")
        cols[1].metric("Protein", f"{meal['nutrition']['protein']} g")
        cols[2].metric("Carbs", f"{meal['nutrition']['carbs']} g")

# 8. UI Section
if not auth_section():
    st.warning("Please login to analyze and save meals.")
    st.stop()

st.title("🍽️ Indian Food Nutrition Analyzer")
st.markdown("Upload an Indian food image and we'll analyze its nutritional content!")

uploaded = st.file_uploader("Upload Food Image", type=['jpg', 'jpeg', 'png'])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        input_mode = st.radio("Select input mode", ["Grams", "Number of pieces"], index=0)
        quantity = st.number_input("Enter Quantity", min_value=1, value=100 if input_mode == "Grams" else 1, step=1)

        if st.button("🤖 Analyze Nutrition"):
            with st.spinner("Analyzing..."):
                food_class, confidence = predict_food(img, model, class_labels)

            if food_class not in ["model_not_loaded", "prediction_error"]:
                st.markdown(f"### 🍽️ Predicted: `{food_class}` ({confidence * 100:.2f}% confidence)")
                st.markdown(f"### 📊 Quantity: `{quantity} {'grams' if input_mode == 'Grams' else 'piece(s)'}`")

                food_key = food_class.lower().strip()
                if food_key in indian_nutrition_db:
                    base = indian_nutrition_db[food_key]
                    unit = base['grams']
                    factor = (quantity / 100) if input_mode == "Grams" else quantity
                    if input_mode == "Grams" and unit != 100:
                        factor = quantity / unit
                    elif input_mode == "Number of pieces" and unit == 100:
                        factor = (quantity * 100) / 100

                    nutrition = {k: round(v * factor, 2) for k, v in base.items() if k != 'grams'}

                    st.success("Nutrition Data (from Local Indian Database)")
                    cols = st.columns(3)
                    cols[0].metric("Calories", f"{nutrition['calories']} kcal")
                    cols[1].metric("Protein", f"{nutrition['protein']} g")
                    cols[2].metric("Carbs", f"{nutrition['carbs']} g")

                    cols = st.columns(3)
                    cols[0].metric("Fat", f"{nutrition['fat']} g")
                    cols[1].metric("Sugar", f"{nutrition['sugar']} g")
                    cols[2].metric("Fiber", f"{nutrition['fiber']} g")

                    log_meal_to_firestore(food_class, nutrition, quantity, input_mode)
                else:
                    st.warning(f"'{food_class}' not found in local database. Please consider adding it manually.")
            else:
                st.error("Food could not be predicted. Try another image.")

show_meal_history()
