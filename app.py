
# import streamlit as st
# import os
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import cv2
# import datetime
# from typing import List

# # Import your firebase_config which must expose `db`
# from utils import firebase_config
# from firebase_admin import auth, firestore

# # rule engine
# import rule_book
# from rule_book import UserProfile, FoodItem as RBFoodItem, evaluate, derive_personal_limits

# # ========== Streamlit config ==========
# st.set_page_config(page_title="Indian Food Nutrition Analyzer", layout="wide")

# # ========== Session defaults ==========
# if 'user' not in st.session_state:
#     st.session_state.user = None
# if 'profile' not in st.session_state:
#     st.session_state.profile = None
# if 'profile_loaded' not in st.session_state:
#     st.session_state.profile_loaded = False

# # ========== Auth UI ==========
# def auth_section():
#     st.sidebar.header("🔐 Login / Signup")
#     if st.session_state.user:
#         st.sidebar.success(f"Logged in as: {st.session_state.user['email']}")
#         if st.sidebar.button("Logout"):
#             st.session_state.user = None
#             st.session_state.profile_loaded = False
#             st.experimental_rerun()
#         return True

#     choice = st.sidebar.radio("Choose", ["Login", "Sign Up"])
#     email = st.sidebar.text_input("Email", key="auth_email")
#     password = st.sidebar.text_input("Password", type="password", key="auth_pass")

#     if choice == "Sign Up":
#         if st.sidebar.button("Create Account"):
#             try:
#                 user = auth.create_user(email=email, password=password)
#                 st.success("Account created — please login.")
#             except Exception as e:
#                 st.error(f"Sign up failed: {e}")

#     if choice == "Login":
#         if st.sidebar.button("Login"):
#             try:
#                 user = auth.get_user_by_email(email)
#                 st.session_state.user = {'email': email, 'uid': user.uid}
#                 st.experimental_rerun()
#             except Exception as e:
#                 st.error(f"Login failed: {e}")
#     return False

# # ========== Model loading (cached) ==========
# @st.cache_resource
# def load_model_and_labels():
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
#         st.error(f"Model load failed: {e}")
#         return None, []

# model, class_labels = load_model_and_labels()

# # ========== Local nutrition DB (unchanged logic, values per 100g unless noted) ==========
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

# # ========== Prediction helper (no confidence shown to user) ==========
# def predict_food_label(img, model, class_labels):
#     if model is None or img is None:
#         return "model_not_loaded"
#     try:
#         pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((224, 224))
#         arr = np.array(pil) / 255.0
#         arr = np.expand_dims(arr, 0)
#         preds = model.predict(arr)[0]
#         idx = int(np.argmax(preds))
#         label = class_labels[idx]
#         return label.replace("_", " ")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
#         return "prediction_error"

# # ========== Build full nutrition (estimates for sodium / sat fat) ==========
# def build_full_nutrition(food_key: str, quantity: float, input_mode: str):
#     base = indian_nutrition_db.get(food_key)
#     if not base:
#         return None
#     unit = base['grams']
#     if input_mode == "Grams":
#         # scale relative to base unit (usually per 100g)
#         factor = quantity / 100.0
#         if unit != 100:
#             factor = quantity / unit
#     else:
#         # pieces — treat 1 piece as 100g unless unit says otherwise
#         factor = quantity if unit == 1 else (quantity * unit) / 100.0

#     calories = round(base['calories'] * factor, 2)
#     protein = round(base['protein'] * factor, 2)
#     carbs = round(base['carbs'] * factor, 2)
#     fat = round(base['fat'] * factor, 2)
#     sugar = round(base['sugar'] * factor, 2)
#     fiber = round(base['fiber'] * factor, 2)
#     grams = round((unit * factor) if unit else (quantity if input_mode == "Grams" else quantity * 100), 1)

#     # conservative sodium & sat fat estimates
#     est_sodium_per_100g = 150  # mg per 100g baseline
#     sodium_mg = int(est_sodium_per_100g * (grams / 100.0))
#     sat_fat_g = round(0.25 * fat, 2)

#     return {
#         "calories": calories,
#         "protein": protein,
#         "carbs_g": carbs,
#         "fat_g": fat,
#         "sugar_g": sugar,
#         "fiber_g": fiber,
#         "sodium_mg": sodium_mg,
#         "sat_fat_g": sat_fat_g,
#         "grams": grams
#     }

# # ========== Tag inference (simple heuristics) ==========
# TAG_HINTS = {
#     "jalebi": ["sweet", "refined_carb"],
#     "gajar ka halwa": ["sweet"],
#     "laddu": ["sweet"],
#     "samosa": ["fried"],
#     "vada pav": ["fried"],
#     "pav bhaji": ["refined_carb"],
#     "puri bhaji": ["fried"],
#     "paratha": ["refined_carb", "fried"],
#     "paneer butter masala": ["fried", "refined_carb"],
#     "pani puri": ["salty"],
#     "bhel puri": ["refined_carb"],
#     "poha": ["refined_carb"],
#     "plain rice": ["refined_carb"],
#     "dosa": ["refined_carb"]
# }

# def infer_tags(name: str):
#     n = name.lower()
#     tags = []
#     for k, v in TAG_HINTS.items():
#         if k in n:
#             tags.extend(v if isinstance(v, list) else [v])
#     return list(set(tags))

# # ========== Firestore helpers ==========
# def log_meal_to_firestore(food_name: str, nutrition: dict, quantity, input_mode: str, tags: List[str]):
#     if not st.session_state.user:
#         return
#     meal_doc = {
#         'food': food_name,
#         'quantity': quantity,
#         'input_mode': input_mode,
#         'nutrition': nutrition,
#         'tags': tags,
#         'timestamp': datetime.datetime.now()
#     }
#     firebase_config.db.collection("users").document(st.session_state.user['uid']).collection("meals").document().set(meal_doc)

# def get_todays_meals_from_firestore():
#     if not st.session_state.user:
#         return []
#     today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
#     q = firebase_config.db.collection("users").document(st.session_state.user['uid']).collection("meals")\
#         .where("timestamp", ">=", today_start)\
#         .order_by("timestamp", direction=firestore.Query.ASCENDING).stream()
#     items = []
#     for d in q:
#         items.append(d.to_dict())
#     return items

# # Convert Firestore meal docs into rule_book FoodItem list
# def build_today_log_rb():
#     raw = get_todays_meals_from_firestore()
#     rb_items = []
#     for r in raw:
#         nut = r.get("nutrition", {})
#         name = r.get("food", "Unknown")
#         grams = float(nut.get("grams", r.get("quantity", 0)))
#         calories = float(nut.get("calories", 0))
#         carbs_g = float(nut.get("carbs_g", nut.get("carbs", 0)))
#         protein_g = float(nut.get("protein", 0))
#         fat_g = float(nut.get("fat_g", nut.get("fat", 0)))
#         sugar_g = float(nut.get("sugar_g", nut.get("sugar", 0)))
#         fiber_g = float(nut.get("fiber_g", nut.get("fiber", 0)))
#         sodium_mg = float(nut.get("sodium_mg", 0))
#         sat_fat_g = float(nut.get("sat_fat_g", 0))
#         tags = r.get("tags", [])
#         rb = RBFoodItem(
#             name=name,
#             grams=grams,
#             calories=calories,
#             carbs_g=carbs_g,
#             protein_g=protein_g,
#             fat_g=fat_g,
#             sugar_g=sugar_g,
#             fiber_g=fiber_g,
#             sodium_mg=sodium_mg,
#             sat_fat_g=sat_fat_g,
#             trans_fat_g=None,
#             cholesterol_mg=None,
#             tags=tags
#         )
#         rb_items.append(rb)
#     return rb_items

# # ========== Profile UI & normalization ==========
# def normalize_condition_list(lst):
#     if not lst:
#         return []
#     return [s.strip().lower() for s in lst if s and s.strip()]

# def load_profile_ui():
#     st.sidebar.header("👤 Your Profile (used for rules)")
#     with st.sidebar.form("profile_form", clear_on_submit=False):
#         age = st.number_input("Age", min_value=5, max_value=120, value=25, step=1)
#         sex = st.selectbox("Sex", ["male", "female"], index=0)
#         height_cm = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=170.0, step=0.5)
#         weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
#         activity_level = st.selectbox("Daily activity level", ["low", "moderate", "high"], index=1)
#         sleep_hours = st.number_input("Avg sleep hours (optional)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
#         # Present conditions as lowercase options to match rule_book expectations
#         conditions = st.multiselect("Pre-existing conditions", ["diabetes", "hypertension", "obesity", "dyslipidemia"])
#         vegetarian = st.selectbox("Vegetarian?", ["No", "Yes"], index=0)
#         allergies = st.text_input("Allergies (comma separated, optional)")

#         submitted = st.form_submit_button("Save Profile (session)")
#         if submitted:
#             # Normalize conditions and allergies
#             conds = normalize_condition_list(conditions)
#             allergies_list = [a.strip().lower() for a in allergies.split(",") if a.strip()] if allergies else []
#             veg_bool = True if vegetarian == "Yes" else False

#             profile = UserProfile(
#                 user_id=st.session_state.user['uid'] if st.session_state.user else "local",
#                 age=int(age),
#                 sex=sex,
#                 height_cm=float(height_cm),
#                 weight_kg=float(weight_kg),
#                 activity_level=activity_level,
#                 sleep_hours=float(sleep_hours) if sleep_hours else None,
#                 conditions=conds,
#                 vegetarian=veg_bool,
#                 allergies=allergies_list
#             )
#             st.session_state.profile = profile
#             st.session_state.profile_loaded = True
#             st.success("Profile saved for this session. (Conditions normalized)")

# # ========== Meal History display ==========
# def show_meal_history():
#     st.subheader("📅 Today's Meals")
#     meals = get_todays_meals_from_firestore()
#     if not meals:
#         st.info("No meals logged today.")
#         return
#     for m in meals[::-1]:
#         nut = m.get("nutrition", {})
#         st.markdown(f"**{m.get('food','Unknown').title()}** — {m.get('quantity')} {m.get('input_mode')}")
#         cols = st.columns([1,1,1,1])
#         cols[0].write(f"{nut.get('calories','--')} kcal")
#         cols[1].write(f"{nut.get('protein','--')} g P")
#         cols[2].write(f"{nut.get('carbs_g', nut.get('carbs','--'))} g C")
#         cols[3].write(f"{nut.get('fat_g', nut.get('fat','--'))} g F")

# # ========== Main UI ==========
# if not auth_section():
#     st.warning("Please login to use the app (or create an account).")
#     st.stop()

# # Profile area (load/create)
# if not st.session_state.profile_loaded:
#     load_profile_ui()
# else:
#     # quick view
#     with st.sidebar.expander("Profile (view / edit)"):
#         p = st.session_state.profile
#         st.write(f"**Age:** {p.age}  **Sex:** {p.sex}")
#         st.write(f"**Height:** {p.height_cm} cm  **Weight:** {p.weight_kg} kg")
#         st.write(f"**BMI:** {p.bmi}  **Activity:** {p.activity_level}")
#         st.write(f"**Conditions:** {', '.join(p.conditions) if p.conditions else 'None'}")
#         if st.button("Edit Profile"):
#             st.session_state.profile_loaded = False
#             st.experimental_rerun()

# # Activity sliders (affect personalization)
# st.sidebar.header("Activity (per slot)")
# st.sidebar.markdown("Rate 1 (very low) — 100 (very active)")
# morning_act = st.sidebar.slider("Morning", 1, 100, 50, key="m_act")
# afternoon_act = st.sidebar.slider("Afternoon", 1, 100, 50, key="a_act")
# evening_act = st.sidebar.slider("Evening", 1, 100, 30, key="e_act")

# def activity_level_from_avg(avg):
#     if avg < 40: return "low"
#     if avg < 70: return "moderate"
#     return "high"

# avg_act = (morning_act + afternoon_act + evening_act) / 3.0
# if st.session_state.profile:
#     st.session_state.profile.activity_level = activity_level_from_avg(avg_act)
# else:
#     # create default minimal profile (normalized fields)
#     st.session_state.profile = UserProfile(
#         user_id=st.session_state.user['uid'],
#         age=25,
#         sex="male",
#         height_cm=170.0,
#         weight_kg=70.0,
#         activity_level=activity_level_from_avg(avg_act),
#         sleep_hours=7.0,
#         conditions=[],
#         vegetarian=None,
#         allergies=[]
#     )

# # Title and instructions
# st.title("🍽️ Indian Food Nutrition Analyzer — Personalized (Rule-based)")
# st.markdown("Upload a food image, choose quantity and slot; the system recognizes the dish, estimates nutrition from a local DB, then uses the project's rule book to classify the food (Acceptable / Warning / Not Recommended). Model confidence is deliberately not shown to users.")

# # Upload & analyze
# uploaded = st.file_uploader("Upload Food Image", type=['jpg','jpeg','png'])

# col_main, col_side = st.columns([2,1])

# with col_main:
#     if uploaded:
#         file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#         if img is None:
#             st.error("Couldn't decode image. Try another file.")
#         else:
#             st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
#             input_mode = st.radio("Input mode", ["Grams", "Number of pieces"])
#             default_q = 100 if input_mode == "Grams" else 1
#             quantity = st.number_input("Quantity", min_value=1, value=default_q, step=1)
#             meal_slot = st.selectbox("Meal slot", ["Morning", "Afternoon", "Evening"])
#             if st.button("Analyze & Evaluate"):
#                 with st.spinner("Analyzing image and running rule engine..."):
#                     predicted = predict_food_label(img, model, class_labels)

#                 if predicted in ["model_not_loaded", "prediction_error"]:
#                     if predicted == "model_not_loaded":
#                         st.error("Model not loaded; prediction unavailable.")
#                     else:
#                         st.error("Prediction failed. Try another image.")
#                 else:
#                     st.markdown(f"### Predicted: **{predicted.title()}**")
#                     food_key = predicted.lower().strip()
#                     if food_key in indian_nutrition_db:
#                         nutrition = build_full_nutrition(food_key, quantity, input_mode)
#                         if nutrition is None:
#                             st.warning("Nutrition lookup failed for this item.")
#                         else:
#                             st.success("Nutrition (estimated from DB)")
#                             c1, c2, c3 = st.columns(3)
#                             c1.metric("Calories", f"{nutrition['calories']} kcal")
#                             c2.metric("Protein", f"{nutrition['protein']} g")
#                             c3.metric("Carbs", f"{nutrition['carbs_g']} g")
#                             c4, c5, c6 = st.columns(3)
#                             c4.metric("Fat", f"{nutrition['fat_g']} g")
#                             c5.metric("Sugar", f"{nutrition['sugar_g']} g")
#                             c6.metric("Fiber", f"{nutrition['fiber_g']} g")

#                             tags = infer_tags(food_key)
#                             rb_food = RBFoodItem(
#                                 name=predicted.title(),
#                                 grams=nutrition['grams'],
#                                 calories=nutrition['calories'],
#                                 carbs_g=nutrition['carbs_g'],
#                                 protein_g=nutrition['protein'],
#                                 fat_g=nutrition['fat_g'],
#                                 sugar_g=nutrition['sugar_g'],
#                                 fiber_g=nutrition['fiber_g'],
#                                 sodium_mg=nutrition['sodium_mg'],
#                                 sat_fat_g=nutrition['sat_fat_g'],
#                                 trans_fat_g=None,
#                                 cholesterol_mg=None,
#                                 tags=tags
#                             )

#                             # Build today's log items for evaluate()
#                             today_log = build_today_log_rb()

#                             # Evaluate — pass normalized profile
#                             eval_res = evaluate(rb_food, st.session_state.profile, today_log, now=None)

#                             # Present decision & explanations
#                             st.markdown("## Rule-based Evaluation")
#                             if eval_res.category == "Acceptable":
#                                 st.success(f"Decision: **{eval_res.category}**")
#                             elif eval_res.category == "Warning":
#                                 st.warning(f"Decision: **{eval_res.category}**")
#                             else:
#                                 st.error(f"Decision: **{eval_res.category}**")

#                             st.markdown("**Reasons:**")
#                             for r in eval_res.reasons:
#                                 st.write(f"- {r}")

#                             if eval_res.tips:
#                                 st.markdown("**Tips:**")
#                                 for t in eval_res.tips:
#                                     st.write(f"- {t}")

#                             if eval_res.alternatives:
#                                 st.markdown("**Alternatives:**")
#                                 for a in eval_res.alternatives:
#                                     st.write(f"- {a}")

#                             # Log meal (DB structure unchanged; add tags & nutrition extras)
#                             log_meal_to_firestore(predicted.title(), nutrition, quantity, input_mode, tags)
#                             st.success("Meal logged to your account.")
#                     else:
#                         st.warning(f"'{predicted}' not found in local nutrition DB. Consider adding it (future feature).")
#     else:
#         st.info("Upload an image to begin.")

# with col_side:
#     st.header("Profile & Daily Targets")
#     p = st.session_state.profile
#     st.write(f"**BMI:** {p.bmi}")
#     limits = derive_personal_limits(p)
#     st.markdown("**Daily targets / caps**")
#     l1, l2, l3 = st.columns(3)
#     l1.write(f"Calories: **{limits.calorie_target} kcal**")
#     l2.write(f"Protein: **{limits.protein_target_g} g**")
#     l3.write(f"Sugar limit: **{limits.sugar_limit_g} g**")
#     l4, l5, l6 = st.columns(3)
#     l4.write(f"Sodium limit: **{limits.sodium_limit_mg} mg**")
#     l5.write(f"Fiber target: **{limits.fiber_target_g} g**")
#     l6.write(f"Fat cap: **{limits.fat_cap_g} g**")

#     st.markdown("---")
#     show_meal_history()

# st.markdown("---")
# st.caption("Notes: Profile conditions are normalized (lowercase) so rule engine rules like 'diabetes' will match reliably. Model confidence is hidden per spec.")
import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import datetime
from typing import List

from utils import firebase_config
from firebase_admin import auth, firestore

import rule_book
from rule_book import UserProfile, FoodItem as RBFoodItem, evaluate, derive_personal_limits

st.set_page_config(page_title="Indian Food Nutrition Analyzer", layout="wide")

if 'user' not in st.session_state:
    st.session_state.user = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'profile_loaded' not in st.session_state:
    st.session_state.profile_loaded = False

def auth_section():
    st.sidebar.header("🔐 Login / Signup")
    if st.session_state.user:
        st.sidebar.success(f"Logged in as: {st.session_state.user['email']}")
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.profile_loaded = False
            st.experimental_rerun()
        return True

    choice = st.sidebar.radio("Choose", ["Login", "Sign Up"])
    email = st.sidebar.text_input("Email", key="auth_email")
    password = st.sidebar.text_input("Password", type="password", key="auth_pass")

    if choice == "Sign Up":
        if st.sidebar.button("Create Account"):
            try:
                user = auth.create_user(email=email, password=password)
                st.success("Account created — please login.")
            except Exception as e:
                st.error(f"Sign up failed: {e}")

    if choice == "Login":
        if st.sidebar.button("Login"):
            try:
                user = auth.get_user_by_email(email)
                st.session_state.user = {'email': email, 'uid': user.uid}
                # Load profile from Firestore when logging in
                doc = firebase_config.db.collection("users").document(user.uid).get()
                if doc.exists:
                    data = doc.to_dict()
                    st.session_state.profile = UserProfile(
                        user_id=user.uid,
                        age=int(data.get("age", 25)),
                        sex=data.get("sex", "male"),
                        height_cm=float(data.get("height_cm", 170)),
                        weight_kg=float(data.get("weight_kg", 70)),
                        activity_level=data.get("activity_level", "moderate"),
                        sleep_hours=float(data.get("sleep_hours", 7.0)) if data.get("sleep_hours") is not None else None,
                        conditions=data.get("conditions", []),
                        vegetarian=bool(data.get("vegetarian")) if "vegetarian" in data else None,
                        allergies=data.get("allergies", []),
                    )
                    st.session_state.profile_loaded = True
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
    return False

@st.cache_resource
def load_model_and_labels():
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
        st.error(f"Model load failed: {e}")
        return None, []

model, class_labels = load_model_and_labels()

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

def predict_food_label(img, model, class_labels):
    if model is None or img is None:
        return "model_not_loaded"
    try:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((224, 224))
        arr = np.array(pil) / 255.0
        arr = np.expand_dims(arr, 0)
        preds = model.predict(arr)[0]
        idx = int(np.argmax(preds))
        label = class_labels[idx]
        return label.replace("_", " ")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "prediction_error"

def build_full_nutrition(food_key: str, quantity: float, input_mode: str):
    base = indian_nutrition_db.get(food_key)
    if not base:
        return None
    unit = base['grams']
    if input_mode == "Grams":
        factor = quantity / 100.0
        if unit != 100:
            factor = quantity / unit
    else:
        factor = quantity if unit == 1 else (quantity * unit) / 100.0

    calories = round(base['calories'] * factor, 2)
    protein = round(base['protein'] * factor, 2)
    carbs = round(base['carbs'] * factor, 2)
    fat = round(base['fat'] * factor, 2)
    sugar = round(base['sugar'] * factor, 2)
    fiber = round(base['fiber'] * factor, 2)
    grams = round((unit * factor) if unit else (quantity if input_mode == "Grams" else quantity * 100), 1)

    est_sodium_per_100g = 150 
    sodium_mg = int(est_sodium_per_100g * (grams / 100.0))
    sat_fat_g = round(0.25 * fat, 2)

    return {
        "calories": calories,
        "protein": protein,
        "carbs_g": carbs,
        "fat_g": fat,
        "sugar_g": sugar,
        "fiber_g": fiber,
        "sodium_mg": sodium_mg,
        "sat_fat_g": sat_fat_g,
        "grams": grams
    }

TAG_HINTS = {
    "jalebi": ["sweet", "refined_carb"],
    "gajar ka halwa": ["sweet"],
    "laddu": ["sweet"],
    "samosa": ["fried"],
    "vada pav": ["fried"],
    "pav bhaji": ["refined_carb"],
    "puri bhaji": ["fried"],
    "paratha": ["refined_carb", "fried"],
    "paneer butter masala": ["fried", "refined_carb"],
    "pani puri": ["salty"],
    "bhel puri": ["refined_carb"],
    "poha": ["refined_carb"],
    "plain rice": ["refined_carb"],
    "dosa": ["refined_carb"]
}

def infer_tags(name: str):
    n = name.lower()
    tags = []
    for k, v in TAG_HINTS.items():
        if k in n:
            tags.extend(v if isinstance(v, list) else [v])
    return list(set(tags))

def log_meal_to_firestore(food_name: str, nutrition: dict, quantity, input_mode: str, tags: List[str]):
    if not st.session_state.user:
        return
    meal_doc = {
        'food': food_name,
        'quantity': quantity,
        'input_mode': input_mode,
        'nutrition': nutrition,
        'tags': tags,
        'timestamp': datetime.datetime.now()
    }
    firebase_config.db.collection("users").document(st.session_state.user['uid']).collection("meals").document().set(meal_doc)

def get_todays_meals_from_firestore():
    if not st.session_state.user:
        return []
    today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    q = firebase_config.db.collection("users").document(st.session_state.user['uid']).collection("meals")\
        .where("timestamp", ">=", today_start)\
        .order_by("timestamp", direction=firestore.Query.ASCENDING).stream()
    items = []
    for d in q:
        items.append(d.to_dict())
    return items

def build_today_log_rb():
    raw = get_todays_meals_from_firestore()
    rb_items = []
    for r in raw:
        nut = r.get("nutrition", {})
        name = r.get("food", "Unknown")
        grams = float(nut.get("grams", r.get("quantity", 0)))
        calories = float(nut.get("calories", 0))
        carbs_g = float(nut.get("carbs_g", nut.get("carbs", 0)))
        protein_g = float(nut.get("protein", 0))
        fat_g = float(nut.get("fat_g", nut.get("fat", 0)))
        sugar_g = float(nut.get("sugar_g", nut.get("sugar", 0)))
        fiber_g = float(nut.get("fiber_g", nut.get("fiber", 0)))
        sodium_mg = float(nut.get("sodium_mg", 0))
        sat_fat_g = float(nut.get("sat_fat_g", 0))
        tags = r.get("tags", [])
        rb = RBFoodItem(
            name=name,
            grams=grams,
            calories=calories,
            carbs_g=carbs_g,
            protein_g=protein_g,
            fat_g=fat_g,
            sugar_g=sugar_g,
            fiber_g=fiber_g,
            sodium_mg=sodium_mg,
            sat_fat_g=sat_fat_g,
            trans_fat_g=None,
            cholesterol_mg=None,
            tags=tags
        )
        rb_items.append(rb)
    return rb_items

def normalize_condition_list(lst):
    if not lst:
        return []
    return [s.strip().lower() for s in lst if s and s.strip()]

def load_profile_ui():
    st.sidebar.header("👤 Your Profile (used for rules)")
    with st.sidebar.form("profile_form", clear_on_submit=False):
        age = st.number_input("Age", min_value=5, max_value=120, value=25, step=1)
        sex = st.selectbox("Sex", ["male", "female"], index=0)
        height_cm = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=170.0, step=0.5)
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
        activity_level = st.selectbox("Daily activity level", ["low", "moderate", "high"], index=1)
        sleep_hours = st.number_input("Avg sleep hours (optional)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        conditions = st.multiselect("Pre-existing conditions", ["diabetes", "hypertension", "obesity", "dyslipidemia"])
        vegetarian = st.selectbox("Vegetarian?", ["No", "Yes"], index=0)
        allergies = st.text_input("Allergies (comma separated, optional)")

        submitted = st.form_submit_button("Save Profile (session)")
        if submitted:
            conds = normalize_condition_list(conditions)
            allergies_list = [a.strip().lower() for a in allergies.split(",") if a.strip()] if allergies else []
            veg_bool = True if vegetarian == "Yes" else False
            user_id = st.session_state.user['uid'] if st.session_state.user else "local"
            profile = UserProfile(
                user_id=user_id,
                age=int(age),
                sex=sex,
                height_cm=float(height_cm),
                weight_kg=float(weight_kg),
                activity_level=activity_level,
                sleep_hours=float(sleep_hours) if sleep_hours else None,
                conditions=conds,
                vegetarian=veg_bool,
                allergies=allergies_list
            )
            st.session_state.profile = profile
            st.session_state.profile_loaded = True
            st.success("Profile saved for this session. (Conditions normalized)")
            # Save profile to database for persistent login
            if st.session_state.user:
                firebase_config.db.collection("users").document(user_id).set(profile.__dict__)
                
def show_meal_history():
    st.subheader("📅 Today's Meals")
    meals = get_todays_meals_from_firestore()
    if not meals:
        st.info("No meals logged today.")
        return
    for m in meals[::-1]:
        nut = m.get("nutrition", {})
        st.markdown(f"**{m.get('food','Unknown').title()}** — {m.get('quantity')} {m.get('input_mode')}")
        cols = st.columns([1,1,1,1])
        cols[0].write(f"{nut.get('calories','--')} kcal")
        cols[1].write(f"{nut.get('protein','--')} g P")
        cols[2].write(f"{nut.get('carbs_g', nut.get('carbs','--'))} g C")
        cols[3].write(f"{nut.get('fat_g', nut.get('fat','--'))} g F")

if not auth_section():
    st.warning("Please login to use the app (or create an account).")
    st.stop()

if not st.session_state.profile_loaded:
    load_profile_ui()
else:
    with st.sidebar.expander("Profile (view / edit)"):
        p = st.session_state.profile
        st.write(f"**Age:** {p.age}  **Sex:** {p.sex}")
        st.write(f"**Height:** {p.height_cm} cm  **Weight:** {p.weight_kg} kg")
        st.write(f"**BMI:** {p.bmi}  **Activity:** {p.activity_level}")
        st.write(f"**Conditions:** {', '.join(p.conditions) if p.conditions else 'None'}")
        if st.button("Edit Profile"):
            st.session_state.profile_loaded = False
            st.experimental_rerun()

st.sidebar.header("Activity (per slot)")
st.sidebar.markdown("Rate 1 (very low) — 100 (very active)")
morning_act = st.sidebar.slider("Morning", 1, 100, 50, key="m_act")
afternoon_act = st.sidebar.slider("Afternoon", 1, 100, 50, key="a_act")
evening_act = st.sidebar.slider("Evening", 1, 100, 30, key="e_act")

def activity_level_from_avg(avg):
    if avg < 40: return "low"
    if avg < 70: return "moderate"
    return "high"

avg_act = (morning_act + afternoon_act + evening_act) / 3.0
if st.session_state.profile:
    st.session_state.profile.activity_level = activity_level_from_avg(avg_act)
else:
    st.session_state.profile = UserProfile(
        user_id=st.session_state.user['uid'],
        age=25,
        sex="male",
        height_cm=170.0,
        weight_kg=70.0,
        activity_level=activity_level_from_avg(avg_act),
        sleep_hours=7.0,
        conditions=[],
        vegetarian=None,
        allergies=[]
    )

st.title("🍽️ Indian Food Nutrition Analyzer — Personalized (Rule-based)")
st.markdown("Upload a food image, choose quantity and slot; the system recognizes the dish, estimates nutrition from a local DB, then uses the project's rule book to classify the food (Acceptable / Warning / Not Recommended). Model confidence is deliberately not shown to users.")

uploaded = st.file_uploader("Upload Food Image", type=['jpg','jpeg','png'])

col_main, col_side = st.columns([2,1])

with col_main:
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Couldn't decode image. Try another file.")
        else:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
            input_mode = st.radio("Input mode", ["Grams", "Number of pieces"])
            default_q = 100 if input_mode == "Grams" else 1
            quantity = st.number_input("Quantity", min_value=1, value=default_q, step=1)
            meal_slot = st.selectbox("Meal slot", ["Morning", "Afternoon", "Evening"])
            if st.button("Analyze & Evaluate"):
                with st.spinner("Analyzing image and running rule engine..."):
                    predicted = predict_food_label(img, model, class_labels)

                if predicted in ["model_not_loaded", "prediction_error"]:
                    if predicted == "model_not_loaded":
                        st.error("Model not loaded; prediction unavailable.")
                    else:
                        st.error("Prediction failed. Try another image.")
                else:
                    st.markdown(f"### Predicted: **{predicted.title()}**")
                    food_key = predicted.lower().strip()
                    if food_key in indian_nutrition_db:
                        nutrition = build_full_nutrition(food_key, quantity, input_mode)
                        if nutrition is None:
                            st.warning("Nutrition lookup failed for this item.")
                        else:
                            st.success("Nutrition (estimated from DB)")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Calories", f"{nutrition['calories']} kcal")
                            c2.metric("Protein", f"{nutrition['protein']} g")
                            c3.metric("Carbs", f"{nutrition['carbs_g']} g")
                            c4, c5, c6 = st.columns(3)
                            c4.metric("Fat", f"{nutrition['fat_g']} g")
                            c5.metric("Sugar", f"{nutrition['sugar_g']} g")
                            c6.metric("Fiber", f"{nutrition['fiber_g']} g")

                            tags = infer_tags(food_key)
                            rb_food = RBFoodItem(
                                name=predicted.title(),
                                grams=nutrition['grams'],
                                calories=nutrition['calories'],
                                carbs_g=nutrition['carbs_g'],
                                protein_g=nutrition['protein'],
                                fat_g=nutrition['fat_g'],
                                sugar_g=nutrition['sugar_g'],
                                fiber_g=nutrition['fiber_g'],
                                sodium_mg=nutrition['sodium_mg'],
                                sat_fat_g=nutrition['sat_fat_g'],
                                trans_fat_g=None,
                                cholesterol_mg=None,
                                tags=tags
                            )

                            today_log = build_today_log_rb()
                            eval_res = evaluate(rb_food, st.session_state.profile, today_log, now=None)

                            st.markdown("## Rule-based Evaluation")
                            if eval_res.category == "Acceptable":
                                st.success(f"Decision: **{eval_res.category}**")
                            elif eval_res.category == "Warning":
                                st.warning(f"Decision: **{eval_res.category}**")
                            else:
                                st.error(f"Decision: **{eval_res.category}**")

                            st.markdown("**Reasons:**")
                            for r in eval_res.reasons:
                                st.write(f"- {r}")
                            if eval_res.tips:
                                st.markdown("**Tips:**")
                                for t in eval_res.tips:
                                    st.write(f"- {t}")
                            if eval_res.alternatives:
                                st.markdown("**Alternatives:**")
                                for a in eval_res.alternatives:
                                    st.write(f"- {a}")
                            log_meal_to_firestore(predicted.title(), nutrition, quantity, input_mode, tags)
                            st.success("Meal logged to your account.")
                    else:
                        st.warning(f"'{predicted}' not found in local nutrition DB. Consider adding it (future feature).")
    else:
        st.info("Upload an image to begin.")

with col_side:
    st.header("Profile & Daily Targets")
    p = st.session_state.profile

    today_log = build_today_log_rb()
    totals = rule_book.sum_daily_totals(today_log)
    limits = derive_personal_limits(p)
    calories_used = getattr(totals, 'calories', 0.0)
    calories_remaining = max(0, round(limits.calorie_target - calories_used, 1))

    st.write(f"**BMI:** {p.bmi}")
    st.markdown("**Daily targets / caps**")
    l1, l2, l3 = st.columns(3)
    l1.write(f"Calories: **{limits.calorie_target} kcal**")
    l2.write(f"Consumed: **{calories_used} kcal**")
    l3.write(f"Remaining: **{calories_remaining} kcal**")
    l4, l5, l6 = st.columns(3)
    l4.write(f"Sodium limit: **{limits.sodium_limit_mg} mg**")
    l5.write(f"Protein: **{limits.protein_target_g} g**")
    l6.write(f"Sugar limit: **{limits.sugar_limit_g} g**")
    l7, l8 = st.columns(2)
    l7.write(f"Fiber target: **{limits.fiber_target_g} g**")
    l8.write(f"Fat cap: **{limits.fat_cap_g} g**")

    st.markdown("---")
    show_meal_history()

st.markdown("---")
st.caption("Notes: Profile conditions are normalized (lowercase) so rule engine rules like 'diabetes' will match reliably. Model confidence is hidden per spec.")
