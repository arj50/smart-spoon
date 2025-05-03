import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import re

# Load the trained model
model = load_model('food_model.h5')

# Class names
class_names = ['aloo_gobi', 'aloo_tikki', 'biriyani', 'butter_chicken', 
               'chana_masala', 'chicken_razala', 'chicken_tikka', 'gulab_jamun', 
               'palak_paneer', 'paneer_butter_masala']

# Nutritional info
food_info = {
    
    'aloo_gobi': {
        'description': "Aloo Gobi is a vegetarian dish originating from the Indian subcontinent, made with potatoes (aloo) and cauliflower (gobi) cooked together with onions, tomatoes, and a blend of aromatic spices like turmeric, cumin, ginger, and garlic.",
        'nutrition': {
            'Calories': '150 - 250 kcal (can be higher depending on oil)',
            'Sodium': '~ 600 - 900 mg (highly variable based on added salt)',
            'Fat': '~ 10 - 20 g (primarily from cooking oil/ghee)',
            'Protein': '~ 4 - 7 g',
            'Carbohydrates': '~ 20 - 30 g (from potatoes and cauliflower)',
            'Fiber': '~ 5 - 8 g (good source from cauliflower)',
            'Note': 'These values are estimates. Actual nutrition depends heavily on the recipe, portion size, and amount of oil used.'
        }
    },
    'aloo_tikki': {
        'description': "Aloo Tikki is a popular North Indian snack made from boiled potatoes, mashed and seasoned with various spices, sometimes with fillings like peas or lentils. These patties are typically pan-fried or deep-fried until golden brown and crispy.",
        'nutrition': {
            'Calories': '200 - 350 kcal (significantly higher if deep-fried)',
            'Sodium': '~ 300 - 600 mg (variable based on added salt and spices)',
            'Fat': '~ 10 - 25 g (primarily from cooking oil)',
            'Protein': '~ 3 - 5 g',
            'Carbohydrates': '~ 25 - 40 g (mainly from potatoes)',
            'Fiber': '~ 2 - 4 g',
            'Note': 'These values are estimates. Frying method and ingredients greatly impact the fat and calorie content.'
        }
    },
    'biriyani': {
        'description': "Biriyani is a fragrant mixed rice dish originating from the Indian subcontinent, made with basmati rice, spices, meat (chicken, mutton, fish, or prawns) or vegetables, yogurt, onions, and often saffron. It's typically cooked in layers, allowing the flavors to meld together.",
        'nutrition': {
            'Calories': '300 - 500 kcal (depending on meat content and oil)',
            'Sodium': '~ 700 - 1200 mg (variable based on added salt and spices)',
            'Fat': '~ 15 - 30 g (from meat, cooking oil/ghee, and nuts)',
            'Protein': '~ 15 - 25 g (depending on the amount and type of meat)',
            'Carbohydrates': '~ 30 - 50 g (mainly from rice)',
            'Fiber': '~ 2 - 5 g (from vegetables and spices)',
            'Note': 'These values are estimates and vary greatly with the type of biriyani (vegetable, meat), and the amount of oil and meat used.'
        }
    },
    'butter_chicken': {
        'description': "Butter Chicken, also known as Murgh Makhani, is a creamy and rich Indian dish made with chicken cooked in a tomato-based sauce with butter, cream, and a variety of spices.",
        'nutrition': {
            'Calories': '350 - 550 kcal (high due to cream and butter)',
            'Sodium': '~ 600 - 1000 mg (variable based on added salt and spices)',
            'Fat': '~ 25 - 40 g (primarily from butter, cream, and cooking oil)',
            'Protein': '~ 20 - 30 g (from chicken)',
            'Carbohydrates': '~ 10 - 20 g (from the gravy)',
            'Fiber': '~ 1 - 3 g',
            'Note': 'These values are estimates. The richness of the gravy significantly impacts the fat and calorie content.'
        }
    },
    'chana_masala': {
        'description': "Chana Masala is a flavorful and tangy North Indian curry made from chickpeas (chana) cooked in a tomato and onion-based gravy with a blend of aromatic spices like cumin, coriander, turmeric, and garam masala.",
        'nutrition': {
            'Calories': '200 - 350 kcal (depending on oil content)',
            'Sodium': '~ 500 - 900 mg (variable based on added salt and spices)',
            'Fat': '~ 8 - 15 g (primarily from cooking oil)',
            'Protein': '~ 10 - 15 g (good source of plant-based protein from chickpeas)',
            'Carbohydrates': '~ 30 - 50 g (mainly from chickpeas)',
            'Fiber': '~ 10 - 15 g (excellent source of fiber from chickpeas)',
            'Note': 'These values are estimates. The amount of oil used in cooking will affect the calorie and fat content.'
        }
    },
    'chicken_razala': {
        'description': "Chicken Razala is a rich and aromatic Mughlai-style chicken curry known for its creamy texture and the use of nuts, seeds, and fragrant spices like cardamom and kewra water. It typically has a lighter color compared to other chicken curries.",
        'nutrition': {
            'Calories': '300 - 500 kcal (due to cream, nuts, and seeds)',
            'Sodium': '~ 500 - 900 mg (variable based on added salt and spices)',
            'Fat': '~ 20 - 35 g (primarily from cream, nuts, seeds, and cooking oil)',
            'Protein': '~ 20 - 30 g (from chicken)',
            'Carbohydrates': '~ 8 - 15 g (from the gravy)',
            'Fiber': '~ 1 - 3 g (from nuts and spices)',
            'Note': 'These values are estimates. The use of rich ingredients makes it higher in fat and calories.'
        }
    },
    'chicken_tikka': {
        'description': "Chicken Tikka is a popular dish made with boneless chicken marinated in yogurt and spices, then grilled or baked until tender and smoky.",
        'nutrition': {
            'Calories': '250 - 400 kcal (depending on marinade and cooking method)',
            'Sodium': '~ 400 - 700 mg (variable based on marinade ingredients)',
            'Fat': '~ 10 - 25 g (from yogurt, marinade, and cooking oil if used)',
            'Protein': '~ 25 - 35 g (excellent source of lean protein)',
            'Carbohydrates': '~ 5 - 10 g (from yogurt and marinade)',
            'Fiber': '~ 1 - 2 g',
            'Note': 'These values are estimates. Grilling or baking reduces fat compared to frying.'
        }
    },
    'gulab_jamun': {
        'description': "Gulab Jamun is a sweet dessert made from deep-fried milk solids (khoya or mawa) soaked in sugar syrup flavored with rose water or cardamom.",
        'nutrition': {
            'Calories': '300 - 450 kcal (high in sugar and fat)',
            'Sodium': '~ 50 - 100 mg (naturally occurring in milk solids)',
            'Fat': '~ 15 - 25 g (from deep-frying)',
            'Protein': '~ 3 - 5 g',
            'Carbohydrates': '~ 40 - 60 g (primarily from sugar)',
            'Fiber': '~ 0 - 1 g',
            'Note': 'These values are estimates. As a dessert, it\'s high in calories, sugar, and fat.'
        }
    },
    'palak_paneer': {
        'description': "Palak Paneer is a popular North Indian vegetarian dish made with spinach (palak) and Indian cheese (paneer), cooked together with onions, garlic, ginger, and a blend of spices.",
        'nutrition': {
            'Calories': '250 - 400 kcal (depending on cream and oil content)',
            'Sodium': '~ 500 - 800 mg (variable based on added salt and spices)',
            'Fat': '~ 15 - 25 g (primarily from paneer and cooking oil/cream)',
            'Protein': '~ 15 - 20 g (good source of protein from paneer and spinach)',
            'Carbohydrates': '~ 10 - 15 g (from spinach and other vegetables)',
            'Fiber': '~ 3 - 5 g (good source of fiber from spinach)',
            'Note': 'These values are estimates. The use of cream can significantly increase the fat and calorie content.'
        }
    },
    'paneer_butter_masala': {
        'description': "Paneer Butter Masala is a rich and creamy North Indian curry made with Indian cheese (paneer) cooked in a smooth tomato-based gravy, enriched with butter and cream, and flavored with aromatic spices.",
        'nutrition': {
            'Calories': '350 - 550 kcal (high due to butter and cream)',
            'Sodium': '~ 600 - 1000 mg (variable based on added salt and spices)',
            'Fat': '~ 25 - 40 g (primarily from butter, cream, and paneer)',
            'Protein': '~ 15 - 20 g (from paneer)',
            'Carbohydrates': '~ 10 - 15 g (from the gravy)',
            'Fiber': '~ 1 - 3 g',
            'Note': 'These values are estimates. The richness of the gravy makes it high in fat and calories.'
        }
    }
}



# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# Parse sodium mg from string
def parse_sodium(s):
    nums = re.findall(r"(\d+)", s)
    nums = list(map(int, nums))
    return sum(nums)/len(nums)*1 if nums else 0

# Build sodium lookup
sodium_lookup = {f: parse_sodium(info['nutrition']['Sodium']) for f, info in food_info.items()}

# Sidebar navigation
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Choose Function", ["Food Recognition", "Low-Sodium Diet Plan"])

if page == "Food Recognition":
    st.title('Indian Food Recognition and Nutrition')
    uploaded = st.file_uploader("Upload food image", type=["jpg","png","jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)
        if st.button('Predict'):
            arr = preprocess_image(img)
            pred = model.predict(arr)
            cls = class_names[np.argmax(pred)]
            conf = 100*np.max(pred)
            st.write(f"**Predicted**: {cls} ({conf:.2f}%)")
            info = food_info[cls]
            st.write("**Description:**", info['description'])
            
            # Nutritional Information Display
            st.subheader("Nutritional Information (per serving):")
            for nutri_name, nutri_val in info['nutrition'].items():
                st.write(f"- **{nutri_name}**: {nutri_val}")


elif page == "Low-Sodium Diet Plan":
    st.title('Low-Sodium Diet Recommendations')
    threshold = st.slider('Max Sodium (mg)', min_value=0, max_value=1200, value=600, step=50)
    st.write(f"Showing foods with sodium ≤ {threshold} mg")
    # Filter foods
    recs = [(f, food_info[f]['nutrition']['Sodium'], sodium_lookup[f]) \
            for f in sodium_lookup if sodium_lookup[f] <= threshold]
    if recs:
        for name, sodium_str, sodium_val in sorted(recs, key=lambda x: x[2]):
            st.subheader(name.replace('_', ' ').title())
            st.write(f"Sodium: {sodium_str}")
            st.write(food_info[name]['description'])
    else:
        st.write("No foods match that sodium threshold.")
