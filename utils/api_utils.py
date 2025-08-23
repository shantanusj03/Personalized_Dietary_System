import requests
import os
from dotenv import load_dotenv
import streamlit as st
from functools import lru_cache

load_dotenv()

def validate_credentials():
    """Helper to validate credentials before API calls"""
    app_id = os.getenv('EDAMAM_APP_ID')
    app_key = os.getenv('EDAMAM_APP_KEY')
    
    if not app_id or not app_key:
        st.error("""
        ❌ Missing API credentials. Please:
        1. Get keys from Edamam dashboard
        2. Add them to .env file
        3. Restart the app
        """)
        return False
    return True

@lru_cache(maxsize=100)
def get_nutrition_data(food_name):
    """Robust API handler with multiple fallback strategies"""
    if not validate_credentials():
        return None
    
    try:
        # Standardize food name input
        query = f"100g {food_name.lower().strip().split()[0]}" if not food_name[0].isdigit() else food_name
        
        response = requests.get(
            "https://api.edamam.com/api/nutrition-data",
            params={
                'app_id': os.getenv('EDAMAM_APP_ID'),
                'app_key': os.getenv('EDAMAM_APP_KEY'),
                'ingr': query
            },
            timeout=15
        )
        
        # Handle API errors
        if response.status_code == 401:
            st.error("🔑 Invalid API credentials - check your .env file")
            return None
        elif response.status_code != 200:
            st.warning(f"⚠️ API Error {response.status_code} - Service may be down")
            return None
            
        data = response.json()
        
        # Handle new API response format
        if 'ingredients' in data and data['ingredients']:
            if 'parsed' in data['ingredients'][0] and data['ingredients'][0]['parsed']:
                nutrients = data['ingredients'][0]['parsed'][0].get('nutrients', {})
                return {
                    'calories': nutrients.get('ENERC_KCAL', {}).get('quantity', 0),
                    'carbs': nutrients.get('CHOCDF', {}).get('quantity', 0),
                    'protein': nutrients.get('PROCNT', {}).get('quantity', 0),
                    'fat': nutrients.get('FAT', {}).get('quantity', 0),
                    'sugar': nutrients.get('SUGAR', {}).get('quantity', 0),
                    'sodium': nutrients.get('NA', {}).get('quantity', 0),
                    'fiber': nutrients.get('FIBTG', {}).get('quantity', 0)
                }
        
        # Fallback to old response format check
        required_keys = ['calories', 'totalNutrients']
        if all(k in data for k in required_keys):
            return {
                'calories': data.get('calories', 0),
                'carbs': data['totalNutrients'].get('CHOCDF', {}).get('quantity', 0),
                'protein': data['totalNutrients'].get('PROCNT', {}).get('quantity', 0),
                'fat': data['totalNutrients'].get('FAT', {}).get('quantity', 0),
                'sugar': data['totalNutrients'].get('SUGAR', {}).get('quantity', 0),
                'sodium': data['totalNutrients'].get('NA', {}).get('quantity', 0),
                'fiber': data['totalNutrients'].get('FIBTG', {}).get('quantity', 0)
            }
            
        st.warning("⚠️ Unexpected API response format")
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Network Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"⚠️ Unexpected Error: {str(e)}")
        return None

def check_api_status():
    """Check if Edamam API is operational"""
    try:
        response = requests.get(
            "https://api.edamam.com/api/nutrition-data",
            params={
                'app_id': os.getenv('EDAMAM_APP_ID').strip(),
                'app_key': os.getenv('EDAMAM_APP_KEY').strip(),
                'ingr': '1 banana'
            },
            timeout=5
        )
        return response.status_code == 200
    except:
        return False