import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import os

def master_train():
    # 1. Load Original Data
    print("Step 1: Loading Original CSV...")
    if not os.path.exists('crime_dataset_india.csv'):
        print("❌ Error: 'crime_dataset_india.csv' nahi mili!")
        return
    
    df = pd.read_csv('crime_dataset_india.csv')
    df.columns = df.columns.str.strip()

    # 2. Hardcoded Cities (Mapping)
    india_cities_coords = {
        'Ahmedabad': [23.0225, 72.5714], 'Chennai': [13.0827, 80.2707],
        'Ludhiana': [30.9010, 75.8573], 'Pune': [18.5204, 73.8567],
        'Delhi': [28.6139, 77.2090], 'Mumbai': [19.0760, 72.8777],
        'Surat': [21.1702, 72.8311], 'Visakhapatnam': [17.6868, 83.2185],
        'Bangalore': [12.9716, 77.5946], 'Kolkata': [22.5726, 88.3639],
        'Ghaziabad': [28.6692, 77.4538], 'Hyderabad': [17.3850, 78.4867],
        'Jaipur': [26.9124, 75.7873], 'Lucknow': [26.8467, 80.9462],
        'Bhopal': [23.2599, 77.4126], 'Patna': [25.5941, 85.1376],
        'Kanpur': [26.4499, 80.3319], 'Varanasi': [25.3176, 82.9739],
        'Nagpur': [21.1458, 79.0882], 'Meerut': [28.9845, 77.7064],
        'Thane': [19.2183, 72.9781], 'Indore': [22.7196, 75.8577],
        'Rajkot': [22.3039, 70.8022], 'Vasai': [19.3919, 72.8397],
        'Agra': [27.1767, 78.0081], 'Kalyan': [19.2403, 73.1305],
        'Nashik': [19.9975, 73.7898], 'Srinagar': [34.0837, 74.7973],
        'Faridabad': [28.4089, 77.3178]
    }

    print("Step 2: Cleaning and Mapping...")
    df['City'] = df['City'].astype(str).str.strip().str.title()
    df['latitude'] = df['City'].map(lambda x: india_cities_coords.get(x, [np.nan, np.nan])[0])
    df['longitude'] = df['City'].map(lambda x: india_cities_coords.get(x, [np.nan, np.nan])[1])

    # Time Extraction
    def get_hour(t):
        try: return int(str(t).split(':')[0][:2])
        except: return 0
    df['hour'] = df['Time of Occurrence'].apply(get_hour)
    df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0).astype('int8')

    # Drop NaNs
    df = df.dropna(subset=['latitude', 'longitude'])
    
    print(f"✅ Data Ready! Total rows for training: {len(df)}")
    
    if len(df) == 0:
        print("❌ Error: Data processing ke baad 0 rows bachi hain. CSV columns check karo!")
        return

    # 3. Training
    df['city_code'] = df['City'].astype('category').cat.codes
    X = df[['city_code', 'hour', 'is_night', 'latitude', 'longitude']]
    y = np.ones(len(df))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Step 3: Training XGBoost...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist')
    model.fit(X_train, y_train)

    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/yss_crime_model.pkl')
    city_mapping = dict(enumerate(df['City'].astype('category').cat.categories))
    joblib.dump(city_mapping, 'models/city_mapping.pkl')

    print("🚀 ALL DONE! Model saved successfully.")

if __name__ == "__main__":
    master_train()