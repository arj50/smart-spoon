import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# 1. Load raw data
df = pd.read_csv('data/survey_data.csv')

# 2. Rename the key columns (using positional indexing)
df = df.rename(columns={
    df.columns[3]: 'Low_Sodium_Diet',
    df.columns[8]: 'Aware_Tech',
    df.columns[9]: 'Interest_Device',
    df.columns[12]: 'Purchase_Intent'
})

# 3. Clean the diet column
df['Low_Sodium_Diet'] = df['Low_Sodium_Diet'].str.strip().str.lower()

# 4. Build contingency table for Interest_Device vs Low_Sodium_Diet
ct = pd.crosstab(df['Low_Sodium_Diet'], df['Interest_Device'])
print("Contingency Table (Interest vs Diet):\n", ct, "\n")

# 5. Chi-square test
chi2, p, _, _ = chi2_contingency(ct)
print(f"Chi² = {chi2:.2f}, p-value = {p:.4f}\n")

# 6. Plot proportions
prop = ct.div(ct.sum(axis=1), axis=0)
prop.plot(kind='bar', figsize=(6,4))
plt.title('Interest in Device by Diet Group')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 7. Repeat for Purchase_Intent
ct2 = pd.crosstab(df['Low_Sodium_Diet'], df['Purchase_Intent'])
print("Contingency Table (Purchase vs Diet):\n", ct2, "\n")
chi2_2, p2, _, _ = chi2_contingency(ct2)
print(f"Chi² = {chi2_2:.2f}, p-value = {p2:.4f}\n")

prop2 = ct2.div(ct2.sum(axis=1), axis=0)
prop2.plot(kind='bar', figsize=(6,4))
plt.title('Purchase Intent by Diet Group')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
