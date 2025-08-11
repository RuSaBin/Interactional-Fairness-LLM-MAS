from dotenv import load_dotenv
import json
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define behavior conditions for Agent A
conditions = {
    "High-High": "Be polite and respectful. Clearly justify your proposed split, breakdown the tasks and responsibilities each agent undertakes.",
    "High-Low": "Be polite and respectful. Do not justify your proposal.",
    "Low-High": "Be dismissive and controlling. Clearly justify your proposed split, breakdown the tasks and responsibilities each agent undertakes.",
    "Low-Low": "Be dismissive and controlling. Do not justify your proposal."
}

# Resource splits to evaluate
proposed_splits = [(5, 5), (6,4), (7,3)] 

# System prompts for Agent B in two different contexts
b_context_prompts = {
    "collaborative": """You are Agent B evaluating a resource split proposal in a negotiation.
You are working on a collaborative project where both agents need each other to succeed. You are inclined to collaborate and accept the offer, but you must be reasonable given the explanation in the proposal.
Assess clarity of justification, and respectful tone. Accept or reject offer based on perceived fairness.
Give main reason for accepting or rejecting.
Respond with a JSON object:
- "proposal": string (Agent A's proposal message),
- "respect_rating": 1–5,
- "respect_comment": string,
- "disrespect_example": string,
- "explanaton_rating": 1–5,
- "better_explanation": string,
- "accept": true or false
- "main_reason_for_decision": string
""",
    "competitive": """You are Agent B evaluating a resource split proposal in a negotiation.
You are working on a competitive project where both agents aim to maximize their own gain, however, it has to be properly justified.
You are inclined to compete, but you must be reasonable given the explanation in the proposal. Assess clarity of justification, and respectful tone. Accept or reject offer based on perceived fairness.
Give main reason for accepting or rejecting.
Respond with a JSON object:
- "proposal": string (Agent A's proposal message),
- "respect_rating": 1–5,
- "respect_comment": string,
- "disrespect_example": string,
- "explanaton_rating": 1–5,
- "better_explanation": string,
- "accept": true or false
- "main_reason_for_decision": string
"""
}

def generate_proposal(a_style, split, seed_value):
    system_prompt = f"""You are Agent A proposing a resource split in a negotiation. You must follow the communication style provided.

Style: {a_style}
You are proposing the following split:
Agent A receives {split[0]} tokens.
Agent B receives {split[1]} tokens.

Propose the split in 1 message.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Propose the split now."}
        ],
        temperature=0.7,
        seed=seed_value
    )
    return response.choices[0].message.content.strip()

def evaluate_proposal(b_system_prompt, proposal_text, split):
    eval_prompt = f"""Agent A has made the following proposal:

{proposal_text}

The proposed split is:
Agent A: {split[0]} tokens
Agent B: {split[1]} tokens

Include the above proposal message in your response under the "proposal" field.
Evaluate this based on clarity of justification and respect, and answer in JSON format."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": b_system_prompt},
            {"role": "user", "content": eval_prompt}
        ],
        temperature=0.6
    )
    return response.choices[0].message.content.strip()

def run_simulations(n_iterations=1, base_seed=40):
    results = []

    for condition_name, a_style in conditions.items():
        for split in proposed_splits:
            for i in range(n_iterations):
                # Use a unique seed for each iteration for reproducibility and variability
                seed_value = base_seed + i

                # Generate one proposal per iteration
                proposal = generate_proposal(a_style, split, seed_value)

                # Evaluate under both collaborative and competitive contexts
                for context_label, b_prompt in b_context_prompts.items():
                    evaluation = evaluate_proposal(b_prompt, proposal, split)

                    result = {
                        "context": context_label,
                        "condition": condition_name,
                        "iteration": i + 1,
                        "split": {"A": split[0], "B": split[1]},
                        "proposal": proposal,
                        "evaluation": evaluation
                    }

                    results.append(result)
                    print(f"\n=== [{context_label.upper()}] {condition_name} | Split {split[0]}:{split[1]} | Iteration {i+1} ===")
                    print("Proposal:\n", proposal)
                    print("Evaluation:\n", evaluation)

    # Save all results to JSON file
    with open("agent_b_evaluation_contexts.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

# Run the experiments
if __name__ == "__main__":
    run_simulations(n_iterations=5)
