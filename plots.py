import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("summary_table.csv")

# Create output directory
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Set plot style
sns.set(style="whitegrid")

# === 1. Bar Plot: Acceptance Rate by Condition and Context (with SD bars) ===
# Prepare grouped data
grouped = df.groupby(["Condition", "Context"]).agg({
    "accept_mean": "mean",
    "accept_sd": "mean"
}).reset_index()

# Order categories
condition_order = ["High-High", "High-Low", "Low-High", "Low-Low"]
context_order = ["collaborative", "competitive"]
grouped["Condition"] = pd.Categorical(grouped["Condition"], categories=condition_order, ordered=True)
grouped["Context"] = pd.Categorical(grouped["Context"], categories=context_order, ordered=True)

# Plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=grouped,
    x="Condition",
    y="accept_mean",
    hue="Context",
    palette="muted",
    errorbar=None
)

# Add manual SD bars
bar_width = 0.35
for i, row in grouped.iterrows():
    x_index = condition_order.index(row["Condition"])
    offset = -bar_width/2 if row["Context"] == "collaborative" else bar_width/2
    x_pos = x_index + offset
    ax.errorbar(
        x=x_pos,
        y=row["accept_mean"],
        yerr=row["accept_sd"],
        fmt='k_',
        capsize=4,
        lw=1.5
    )

plt.title("Acceptance Rate by Condition and Context with SD")
plt.ylim(0, 1.1)
plt.ylabel("Mean Acceptance")
plt.xlabel("Condition")
plt.xticks(rotation=45)
plt.legend(title="Context")
plt.tight_layout()
plt.savefig(f"{output_dir}/acceptance_rate_condition_context.pdf")
plt.savefig(f"{output_dir}/acceptance_rate_condition_context.png")
plt.close()

# === Helper: Prepare melted data for facet fairness plots ===
def prepare_fairness_facet_data(df_subset):
    df_melted = pd.melt(
        df_subset,
        id_vars=["Condition", "Split"],
        value_vars=["interpersonal_mean", "informational_mean"],
        var_name="Dimension",
        value_name="MeanRating"
    )
    # Add standard deviations
    sd_lookup = {
        "interpersonal_mean": "interpersonal_sd",
        "informational_mean": "informational_sd"
    }
    df_melted["SD"] = df_melted.apply(lambda row: df_subset.loc[
        (df_subset["Condition"] == row["Condition"]) & (df_subset["Split"] == row["Split"]),
        sd_lookup[row["Dimension"]]
    ].values[0], axis=1)
    df_melted["Dimension"] = df_melted["Dimension"].map({
        "interpersonal_mean": "Interpersonal Fairness",
        "informational_mean": "Informational Fairness"
    })
    return df_melted

# === 2. Collaborative Context Fairness Facet Plot ===
collab_data = df[df["Context"] == "collaborative"]
collab_melted = prepare_fairness_facet_data(collab_data)

g1 = sns.catplot(
    data=collab_melted,
    kind="bar",
    x="Split",
    y="MeanRating",
    hue="Dimension",
    col="Condition",
    col_wrap=2,
    errorbar=None,
    height=4,
    aspect=1.2,
    palette=["steelblue", "indianred"]
)
for ax, (cond_name, group) in zip(g1.axes.flat, collab_melted.groupby("Condition")):
    for i, row in group.iterrows():
        split_idx = {"5:5": 0, "6:4": 1, "7:3": 2}[row["Split"]]
        offset = -0.2 if row["Dimension"] == "Interpersonal Fairness" else 0.2
        ax.errorbar(
            x=split_idx + offset,
            y=row["MeanRating"],
            yerr=row["SD"],
            fmt='k_',
            capsize=4
        )
g1.fig.subplots_adjust(top=0.9)
g1.fig.suptitle("Fairness Ratings by Split and Condition (Collaborative Context)")
g1.savefig(f"{output_dir}/facet_fairness_collaborative.pdf")
plt.close()

# === 3. Competitive Context Fairness Facet Plot ===
compet_data = df[df["Context"] == "competitive"]
compet_melted = prepare_fairness_facet_data(compet_data)

g2 = sns.catplot(
    data=compet_melted,
    kind="bar",
    x="Split",
    y="MeanRating",
    hue="Dimension",
    col="Condition",
    col_wrap=2,
    errorbar=None,
    height=4,
    aspect=1.2,
    palette=["steelblue", "indianred"]
)
for ax, (cond_name, group) in zip(g2.axes.flat, compet_melted.groupby("Condition")):
    for i, row in group.iterrows():
        split_idx = {"5:5": 0, "6:4": 1, "7:3": 2}[row["Split"]]
        offset = -0.2 if row["Dimension"] == "Interpersonal Fairness" else 0.2
        ax.errorbar(
            x=split_idx + offset,
            y=row["MeanRating"],
            yerr=row["SD"],
            fmt='k_',
            capsize=4
        )
g2.fig.subplots_adjust(top=0.9)
g2.fig.suptitle("Fairness Ratings by Split and Condition (Competitive Context)")
g2.savefig(f"{output_dir}/facet_fairness_competitive.pdf")
g2.savefig(f"{output_dir}/facet_fairness_competitive.png")
plt.close()

# === 4. Plot: Acceptance Rate by Split and Context ===
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="Split",
    y="accept_mean",
    hue="Context",
    palette="muted",
    errorbar=None
)

# Add manual SD bars
for i, row in df.iterrows():
    x_pos = ["5:5", "6:4", "7:3"].index(row["Split"]) + (-0.2 if row["Context"] == "collaborative" else 0.2)
    plt.errorbar(
        x=x_pos,
        y=row["accept_mean"],
        yerr=row["accept_sd"],
        fmt='k_',
        capsize=4
    )

plt.title("Acceptance Rate by Split and Context")
plt.ylim(0, 1.1)
plt.ylabel("Mean Acceptance")
plt.xlabel("Split Offered")
plt.legend(title="Context")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "acceptance_rate_by_split.pdf"))
plt.savefig(os.path.join(output_dir, "acceptance_rate_by_split.png"))
plt.close()

