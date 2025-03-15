from fastapi import FastAPI, File, UploadFile, Form
import pandas as pd
import joblib
import io
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime
import numpy as np
import os
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
import sqlite3  # This is already part of Python's standard library
from sqlite3 import Error
import time

load_dotenv()

SECRET_KEY = os.getenv("RAPID_API_KEY1")

app = FastAPI()

# Load the trained model
model = joblib.load("bot_detector.pkl")  # Ensure the model file exists

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache expiration time in seconds (1 day = 86400 seconds)
CACHE_EXPIRATION = 86400

# Database setup
def create_connection():
    try:
        conn = sqlite3.connect("bot_detector_cache.db")
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def setup_database():
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    username TEXT PRIMARY KEY,
                    bot_probability REAL NOT NULL,
                    user_probability REAL NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            ''')
            conn.commit()
            print("Database initialized successfully")
        except Error as e:
            print(f"Database setup error: {e}")
        finally:
            conn.close()
    else:
        print("Error! Cannot create the database connection.")

# Call setup on startup
setup_database()

def get_cached_result(username):
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT username, bot_probability, user_probability, timestamp FROM prediction_cache WHERE username = ?",
                (username,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                # Check if cache is still valid
                current_time = int(time.time())
                if current_time - result[3] <= CACHE_EXPIRATION:
                    return {
                        "id": result[0],
                        "bot_probability": result[1],
                        "user_probability": result[2],
                        "cached": True
                    }
        except Error as e:
            print(f"Cache fetch error: {e}")
            conn.close()
    return None

def cache_result(username, bot_probability, user_probability):
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            current_time = int(time.time())
            cursor.execute(
                "INSERT OR REPLACE INTO prediction_cache (username, bot_probability, user_probability, timestamp) VALUES (?, ?, ?, ?)",
                (username, bot_probability, user_probability, current_time)
            )
            conn.commit()
            conn.close()
        except Error as e:
            print(f"Cache store error: {e}")
            conn.close()

@app.get("/")
async def test():
    return {"msg": "This is the test result"}

@app.post("/predict-csv/")
async def predict_user_csv(file: UploadFile = File(...)):
    try:
        results = []
        # Read file as a pandas dataframe
        df_iter = pd.read_csv(io.StringIO((await file.read()).decode("utf-8")), chunksize=1)

        for chunk in df_iter:
            record_id = chunk.iloc[0, 0]  # First column (assumed as ID)
            features = chunk.iloc[0, 1:].values.reshape(1, -1)  # Remaining columns as features
            
            # Predict using the model
            prediction_proba = model.predict_proba(features)[0]
            bot_probability = prediction_proba[1]  # Probability of bot

            # Convert NumPy types to Python native types
            results.append({"id": str(record_id), "bot_probability": float(bot_probability*100)})

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}

@app.post("/predict-user/")
async def predict_user(username: str = Form(None)):
    try:
        if not username:
            return {"error": f"username {username}"}
            
        # Check cache first
        cached_result = get_cached_result(username)
        if cached_result:
            return cached_result
            
        url = "https://twitter241.p.rapidapi.com/user"
        querystring = {
            "username": username
        }
        headers = {
            "x-rapidapi-key": f"{SECRET_KEY}",
            "x-rapidapi-host": "twitter241.p.rapidapi.com"
        }

        try:
            response = requests.get(url, headers=headers, params=querystring)
            user_data = response.json()["result"]
        except Exception as e:
            return {"error": f"Error fetching user data: {str(e)}"}

        try:
            if "result" in user_data["data"]["user"]:
                user_info = user_data["data"]["user"]["result"]["legacy"]

                favourites_count = user_info["favourites_count"]
                followers_count = user_info["followers_count"]
                friends_count = user_info["friends_count"]
                statuses_count = user_info["statuses_count"]
                verified = user_info["verified"]
                created_at = user_info["created_at"]
                default_profile = int(user_info["default_profile"])
                geo_enabled = int(bool(user_info["location"]))
            else:
                return {"error": "User data not found"}
        except Exception as e:
            return {"error": f"Error parsing user data: {e}"}

        try:
            # Calculate account age in days
            account_created_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
            account_age_days = (datetime.now(account_created_date.tzinfo) - account_created_date).days

            # Calculate average tweets per day
            average_tweets_per_day = statuses_count / account_age_days if account_age_days > 0 else 0

            # Extract values in the specified order
            features = [
                default_profile,
                favourites_count,
                followers_count,
                friends_count,
                geo_enabled,
                statuses_count,
                verified,
                average_tweets_per_day,
                account_age_days
            ]

            # Reshape the features into a 2D array
            features_array = np.array(features).reshape(1, -1)

            # Predict using the model
            prediction_proba = model.predict_proba(features_array)[0]
            bot_probability = prediction_proba[1]  # Probability of bot
            user_probability = prediction_proba[0]  # Probability of User
            
            # Store result in cache
            cache_result(username, float(bot_probability * 100), float(user_probability * 100))
            
            return {
                "id": username,
                "bot_probability": float(bot_probability * 100),
                "user_probability": float(user_probability * 100),
                "cached": False
            }
        except Exception as e:
            return {"error": f"Error in model: {str(e)}"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/cache-stats/")
async def get_cache_stats():
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            # Count total entries
            cursor.execute("SELECT COUNT(*) FROM prediction_cache")
            total_entries = cursor.fetchone()[0]
            
            # Get most recent entries
            cursor.execute(
                "SELECT username, bot_probability, user_probability, timestamp FROM prediction_cache ORDER BY timestamp DESC LIMIT 10"
            )
            recent_entries = [
                {
                    "username": row[0],
                    "bot_probability": row[1],
                    "user_probability": row[2],
                    "cached_at": datetime.fromtimestamp(row[3]).strftime('%Y-%m-%d %H:%M:%S')
                }
                for row in cursor.fetchall()
            ]
            
            conn.close()
            return {
                "total_cached_entries": total_entries,
                "recent_entries": recent_entries
            }
        except Error as e:
            conn.close()
            return {"error": f"Database error: {str(e)}"}
    return {"error": "Database connection failed"}

@app.delete("/clear-cache/")
async def clear_cache():
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_cache")
            conn.commit()
            count = cursor.rowcount
            conn.close()
            return {"message": f"Cache cleared. {count} entries removed."}
        except Error as e:
            conn.close()
            return {"error": f"Failed to clear cache: {str(e)}"}
    return {"error": "Database connection failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)