import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load CSV
df = pd.read_csv("data.csv")

# Apply Apriori
frequent_items = apriori(df, min_support=0.4, use_colnames=True)

# Generate rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6)

print(frequent_items)
print(rules)