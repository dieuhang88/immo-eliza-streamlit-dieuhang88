import pandas as pd

# Load your CSV file

df = pd.read_csv("data/Kangaroo.csv")

# Prepare the data
postcode_locality_df = df[["postCode", "locality"]].dropna()
postcode_locality_df["postCode"] = postcode_locality_df["postCode"].astype(str)
postcode_locality_df = postcode_locality_df.drop_duplicates(subset=["postCode"])

postcode_locality_dict = dict(zip(postcode_locality_df["postCode"], postcode_locality_df["locality"]))

# Create the string content for the .py file
content = "postcode_dict = {\n"
for k, v in postcode_locality_dict.items():
    content += f"    '{k}': '{v}',\n"
content += "}\n"

# Write to a .py file
with open("postcode_dict.py", "w", encoding="utf-8") as f:
    f.write(content)

print("postcode_dict.py has been created!")