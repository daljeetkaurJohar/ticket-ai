from motor.motor_asyncio import AsyncIOMotorClient
import os

client = AsyncIOMotorClient(os.getenv("MONGO_URL"))
db = client[os.getenv("DB_NAME")]
