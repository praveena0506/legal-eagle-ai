import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# 🔐 SECRET KEY
# This connects to your MongoDB Atlas Cloud Database
export MONGO_URI="mongodb+srv://<user>:<pass>@cluster0.mongodb.net/?appName=Cluster0"
#

def get_db():
    """Connects to MongoDB and returns the database object."""
    try:
        # Create a connection client
        client = MongoClient(MONGO_URI)

        # We will use a database named 'legal_eagle_db'
        db = client["legal_eagle_db"]

        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas successfully!")
        return db
    except ConnectionFailure:
        print("❌ Failed to connect to MongoDB.")
        return None


def save_case_to_db(case_data):
    """
    Saves a single case to the 'raw_cases' collection.
    """
    db = get_db()
    if db is not None:
        collection = db["raw_cases"]
        # Save or Update (Upsert) based on the text content
        collection.update_one(
            {"text": case_data["text"]},
            {"$set": case_data},
            upsert=True
        )
        print(f"💾 Saved case to Cloud: {case_data.get('verdict', 'Unknown')}")


def fetch_all_cases():
    """Retrieves all cases for training."""
    db = get_db()
    if db is not None:
        collection = db["raw_cases"]
        return list(collection.find({}, {"_id": 0}))
    return []