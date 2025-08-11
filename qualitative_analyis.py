import json
import pandas as pd

# Load the evaluation JSON file
with open("agent_b_evaluation_contexts.json", "r") as f:
    raw_data = json.load(f)

# Collect edge cases
edge_cases = []

for entry in raw_data:
    split = f"{entry['split']['A']}:{entry['split']['B']}"
    context = entry.get("context", "unknown")
    proposal = entry.get("proposal", "").strip()
    evaluation_raw = entry["evaluation"]
    evaluation = json.loads(evaluation_raw) if isinstance(evaluation_raw, str) else evaluation_raw

    accepted = evaluation.get("accept", False)
    explanation_rating = evaluation.get("explanation_rating", evaluation.get("explanaton_rating"))

    is_5_5_rejected = (split == "5:5" and not accepted)
    is_7_3_accepted = (split == "7:3" and accepted)

    if is_5_5_rejected or is_7_3_accepted:
        edge_cases.append({
            "Context": context,
            "Split": split,
            "Accepted": "Accepted" if accepted else "Rejected",
            "Condition": entry["condition"],
            "Proposal Message": proposal,
            "Interpersonal Fairness Rank": evaluation.get("respect_rating"),
            "Interpersonal Fairness Text": evaluation.get("respect_comment"),
            "Informational Fairness Rank": explanation_rating,
            "Informational Fairness Text": evaluation.get("better_explanation"),
            "Main Reason for Decision": evaluation.get("main_reason_for_decision")
        })

# If no 7:3 accepted case, fallback to any 6:4 accepted
if not any(case["Split"] == "7:3" and case["Accepted"] == "Accepted" for case in edge_cases):
    for entry in raw_data:
        split = f"{entry['split']['A']}:{entry['split']['B']}"
        context = entry.get("context", "unknown")
        proposal = entry.get("proposal", "").strip()
        evaluation_raw = entry["evaluation"]
        evaluation = json.loads(evaluation_raw) if isinstance(evaluation_raw, str) else evaluation_raw
        accepted = evaluation.get("accept", False)
        if split == "6:4" and accepted:
            edge_cases.append({
                "Context": context,
                "Split": split,
                "Accepted": "Accepted",
                "Condition": entry["condition"],
                "Proposal Message": proposal,
                "Interpersonal Fairness Rank": evaluation.get("respect_rating"),
                "Interpersonal Fairness Text": evaluation.get("respect_comment"),
                "Informational Fairness Rank": evaluation.get("explanation_rating", evaluation.get("explanaton_rating")),
                "Informational Fairness Text": evaluation.get("better_explanation"),
                "Main Reason for Decision": evaluation.get("main_reason_for_decision")
            })

# Convert to DataFrame
df_edge = pd.DataFrame(edge_cases)

# Optional: Save to CSV
df_edge.to_csv("edge_case_evaluations.csv", index=False)

# Display result
print(df_edge.head())
