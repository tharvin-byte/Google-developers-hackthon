import pandas as pd
import numpy as np
import os

# Set seed for reproducibility

np.random.seed(42)

# Number of samples

n = 500

# Generate features

gender = np.random.choice(["Male", "Female"], size=n)
income = np.random.randint(1, 16, size=n)        # 1 to 15
experience = np.random.randint(0, 11, size=n)    # 0 to 10
education = np.random.choice(["Low", "Medium", "High"], size=n)

# Map education to numeric weight

edu_map = {"Low": 0, "Medium": 1, "High": 2}
edu_score = np.array([edu_map[e] for e in education])

# ----- Generate y_true (ground truth) -----

base_prob = (
0.3
+ 0.03 * income
+ 0.04 * experience
+ 0.08 * edu_score
)

# Clip probabilities

base_prob = np.clip(base_prob, 0, 1)

y_true = np.random.binomial(1, base_prob)

# ----- Generate y_pred (biased predictions) -----

y_pred = []

for i in range(n):
    prob = base_prob[i]

    # Introduce bias: females get lower probability
    if gender[i] == "Female":
        prob -= 0.15

    # Add small noise
    prob += np.random.normal(0, 0.05)

    # Keep probability valid
    prob = np.clip(prob, 0, 1)

    # Generate prediction
    y_pred.append(np.random.binomial(1, prob))


y_pred = np.array(y_pred)

# Create DataFrame

df = pd.DataFrame({
"gender": gender,
"income": income,
"experience": experience,
"education_level": education,
"y_true": y_true,
"y_pred": y_pred
})

# Save to current directory

file_path = os.path.join(os.getcwd(), "synthetic_fairness_dataset.csv")
df.to_csv(file_path, index=False)

# Output

print("✅ Dataset generated successfully!")
print("📂 Saved at:", file_path)
print("\nSample data:")
print(df.head())
