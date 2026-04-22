# Install dependencies as needed:
# pip install kagglehub[pandas-datasets] python-dotenv
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("KAGGLE_KEY", os.getenv("kaggle_api", ""))

import kagglehub

# Set the path to the file you'd like to load
file_path = "orders.csv"

# Load the latest version
dataset_path = kagglehub.dataset_download("brendanartley/simplifiedinstacartdata")
df = pd.read_csv(os.path.join(dataset_path, file_path))

print("First 5 records:", df.head())

output_path = "orders.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")