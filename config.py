import os

from dotenv import load_dotenv

load_dotenv()

META_API_KEY = os.getenv("META_API_KEY")
META_BASE_URL = "https://api.meta.ai/v1"
