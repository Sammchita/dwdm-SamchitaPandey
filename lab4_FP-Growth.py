import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

# Step 1: Create dataset
data = {
    'Milk':   [1, 0, 1, 1, 0],
    'Bread':  [1, 1, 1, 0, 1],
    'Butter': [0, 1, 1, 1, 0],
    'Cheese': [1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Step 2: Apply FP-Growth
frequent_items = fpgrowth(df, min_support=0.4, use_colnames=True)

# Step 3: Display result
print("Frequent Itemsets:\n", frequent_items)