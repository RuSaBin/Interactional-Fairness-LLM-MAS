import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your data
df = pd.read_csv("summary_table.csv")

# Encode 'Split' as ordinal variable
split_encoding = {'5:5': 1, '6:4': 2, '7:3': 3}
df['split_encoded'] = df['Split'].map(split_encoding)

# Binarize target variable
df['accept'] = (df['accept_mean'] >= 0.5).astype(int)

# Function to fit decision tree, return feature importances and accuracy
def fit_tree_for_context(context):
    df_context = df[df['Context'] == context]
    X = df_context[['split_encoded', 'interpersonal_mean', 'informational_mean']]
    y = df_context['accept']
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    # Predict and compute accuracy on training set
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_,
        'Context': context
    })
    
    print(f"Accuracy for context '{context}': {acc:.2f}")
    
    return importance_df

# Fit models for both contexts
collab_importance = fit_tree_for_context('collaborative')
comp_importance = fit_tree_for_context('competitive')

# Combine results
importance_df = pd.concat([collab_importance, comp_importance], ignore_index=True)

# Display the result
print(importance_df)
