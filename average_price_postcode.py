import pandas as pd
import json

# Example: Load your historical dataset with prices
df = pd.read_csv("data/Kangaroo.csv")  # Make sure this dataset has 'postCode', 'price', 'habitableSurface'

# Calculate price per m²
df['price_per_m2'] = df['price'] / df['habitableSurface']

# Compute average price per m² per postcode
postcode_avg_price = df.groupby('postCode')['price_per_m2'].mean().to_dict()

# Save to JSON file
with open('postcode_avg_price.json', 'w') as f:
    json.dump(postcode_avg_price, f)