#This code is when the minimum support threshold is 30% and minimum support is 2
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
df = pd.read_csv('apriori_pch_dataset.csv')

# Convert items into a list of transactions
transactions = df['Items'].apply(lambda x: x.split(','))

# One-hot encode the data
encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)
encoded_df = pd.DataFrame(encoded_data, columns=encoder.columns_)

# Apply Apriori to find frequent itemsets
frequent_itemsets = apriori(encoded_df, min_support=0.01, use_colnames=True)

# Generate association rules (with the proper arguments)
# Ensure `metric="lift"` and `min_threshold=1.0` are set properly
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Output the results
print(frequent_itemsets.head())
print(rules.head())

