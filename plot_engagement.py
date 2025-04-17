import pandas as pd, numpy as np, joblib, matplotlib.pyplot as plt

df = pd.read_csv("viral_posts.csv")
model = joblib.load("virality_model.joblib")

X = df.drop(columns=["viral_label"])
probs = model.predict_proba(X)[:,1]
df["prob"] = probs
df = df.sort_values("prob", ascending=False)
df["cum_eng"] = df["engagement"].cumsum()
df["cum_pct_eng"] = df["cum_eng"]/df["engagement"].sum()
df["cum_pct_posts"] = np.arange(len(df))/len(df)

# Bootstrap confidence intervals
n_boot = 100  # Number of bootstrap samples
boot_curves = []
for _ in range(n_boot):
    sample = df.sample(frac=1, replace=True)  # Resample with replacement
    sample = sample.sort_values("prob", ascending=False)
    cum_eng = sample["engagement"].cumsum() / sample["engagement"].sum()
    boot_curves.append(cum_eng.values)

boot_curves = np.vstack(boot_curves)
lower = np.percentile(boot_curves, 2.5, axis=0)  # 2.5th percentile
upper = np.percentile(boot_curves, 97.5, axis=0)  # 97.5th percentile

# Plot the engagement curve with confidence intervals
plt.fill_between(df["cum_pct_posts"], lower, upper, alpha=0.3, label="95% CI")
plt.plot(df["cum_pct_posts"], df["cum_pct_eng"], label="Engagement Curve")
plt.xlabel("Proportion of Posts")
plt.ylabel("Cumulative Engagement")
plt.title("Engagement vs. Posts Ranked by Virality Score")
plt.legend()
plt.grid()
plt.show()
# Save the plot 
plt.savefig("engagement_curve.png", dpi=300)