import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("summary_table.csv")

# Encode split as ordinal
split_encoding = {'5:5': 0, '6:4': 1, '7:3': 2}
df['split_encoded'] = df['Split'].map(split_encoding)

# Create binary target
df['accept'] = (df['accept_mean'] >= 0.5).astype(int)

# Function to fit regularized logistic regression and report accuracy
def fit_logistic_with_penalty(context, penalty='l2'):
    df_context = df[df['Context'] == context]
    X = df_context[['split_encoded', 'interpersonal_mean', 'informational_mean']]
    y = df_context['accept']
    
    model = LogisticRegression(penalty=penalty, solver='liblinear', C=1.0)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    print(f"Accuracy for context '{context}' with penalty '{penalty}': {acc:.2f}")
    
    return pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0],
        'Penalty': 'Ridge' if penalty == 'l2' else 'Lasso',
        'Context': context
    })

# Fit for both contexts and both penalties
results = pd.concat([
    fit_logistic_with_penalty('collaborative', penalty='l2'),
    fit_logistic_with_penalty('collaborative', penalty='l1'),
    fit_logistic_with_penalty('competitive', penalty='l2'),
    fit_logistic_with_penalty('competitive', penalty='l1')
], ignore_index=True)

# Display the results
print(results)
