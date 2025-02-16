import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access variables
SECRET_KEY = os.getenv("SECRET_KEY")
API_URL = os.getenv("API_URL")
DEBUG = os.getenv("DEBUG")

print(SECRET_KEY)  # Output: mysecretpassword
print(API_URL)     # Output: https://example.com/api
print(DEBUG)       # Output: True
