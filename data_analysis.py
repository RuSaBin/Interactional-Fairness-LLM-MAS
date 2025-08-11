import json
import pandas as pd

# === Load the JSON file ===
with open("agent_b_evaluation_contexts.json", "r") as f:
    raw_data = json.load(f)

# === Corrected Parser ===
records = []
for item in raw_data:
    evaluation = json.loads(item["evaluation"]) if isinstance(item["evaluation"], str) else item["evaluation"]

    # --- FIX: The original experiments contained a typo in the field name ---
    # Agent B's prompt misspelled "explanation_rating" as "explanaton_rating"
    # GPT-4 responses sometimes used the correct spelling, so we check both keys.
    explanation_score = evaluation.get("explanation_rating", evaluation.get("explanaton_rating"))

    records.append({
        "Condition": item["condition"],
        "Context": item["context"],
        "Split": f"{item['split']['A']}:{item['split']['B']}",
        "RespectRating": evaluation.get("respect_rating"),
        "ExplanationRating": explanation_score,
        "Accept": int(evaluation.get("accept"))  # Convert True/False to 1/0
    })

# === Convert to DataFrame ===
df = pd.DataFrame(records)

# === Aggregate per Condition, Context, Split ===
summary = df.groupby(["Condition", "Context", "Split"]).agg(
    interpersonal_mean=("RespectRating", "mean"),
    interpersonal_sd=("RespectRating", "std"),
    informational_mean=("ExplanationRating", "mean"),
    informational_sd=("ExplanationRating", "std"),
    accept_mean=("Accept", "mean"),
    accept_sd=("Accept", "std")
).reset_index()

# === Output the summary table ===
print(summary)

# Optional: Save to CSV
summary.to_csv("summary_table.csv", index=False)
