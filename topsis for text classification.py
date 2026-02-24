import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# --------------------------------------------------
# Sample Classification Dataset (Binary Sentiment)
# --------------------------------------------------
texts = [
    "I love this product!",
    "This is the worst experience ever.",
    "Amazing quality and great service.",
    "I am very disappointed.",
    "Absolutely fantastic!",
    "Not good at all.",
    "Very happy with the results.",
    "Terrible customer support.",
    "Highly recommend it.",
    "Waste of money."
]

labels = [1,0,1,0,1,0,1,0,1,0]   # 1 = Positive, 0 = Negative

# --------------------------------------------------
# Models for Classification
# --------------------------------------------------
model_list = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/bert-base-uncased-SST-2",
    "cardiffnlp/twitter-roberta-base-sentiment",
]

device = "cpu"
results = []

# --------------------------------------------------
# TOPSIS Functions (Same Logic)
# --------------------------------------------------
def validate_inputs(weights_str, impacts_str, cols):
    weights = weights_str.split(',')
    impacts = impacts_str.split(',')

    if len(weights) != cols or len(impacts) != cols:
        raise ValueError("Weights and impacts must match column count.")

    if any(i not in ['+', '-'] for i in impacts):
        raise ValueError("Impacts must be '+' or '-'.")

    return np.array(weights, dtype=float), impacts


def apply_topsis(matrix, weights, impacts):
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weighted = norm_matrix * weights

    ideal_best = []
    ideal_worst = []

    for i, sign in enumerate(impacts):
        if sign == '+':
            ideal_best.append(weighted[:, i].max())
            ideal_worst.append(weighted[:, i].min())
        else:
            ideal_best.append(weighted[:, i].min())
            ideal_worst.append(weighted[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    scores = dist_worst / (dist_best + dist_worst)
    ranks = scores.argsort()[::-1] + 1

    return scores, ranks

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
def evaluate_model(model_name):
    print(f"\nEvaluating: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    predictions = []
    total_time = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)

        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.time() - start

        total_time += elapsed
        pred = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(pred)

    avg_time = total_time / len(texts)
    accuracy = accuracy_score(labels, predictions)
    # Changed 'f1' score calculation to handle potential multiclass predictions
    # from models while comparing against binary true labels.
    f1 = f1_score(labels, predictions, average='weighted')

    print(f"Accuracy       = {accuracy:.4f}")
    print(f"Avg Time       = {avg_time:.4f}")
    print(f"F1 Score       = {f1:.4f}")

    return accuracy, avg_time, f1


# --------------------------------------------------
# Run Evaluation
# --------------------------------------------------
for model_name in model_list:
    results.append(evaluate_model(model_name))

decision_matrix = np.array(results)

# --------------------------------------------------
# Apply TOPSIS Ranking
# --------------------------------------------------
weights_input = "1,1,1"
impacts_input = "+,-,+"

weights, impacts = validate_inputs(weights_input, impacts_input, decision_matrix.shape[1])
scores, ranks = apply_topsis(decision_matrix, weights, impacts)

best_index = np.argmax(scores)

print("\nTOPSIS Scores:", scores)
print("Best Model:", model_list[best_index])

# --------------------------------------------------
# Create Results Table
# --------------------------------------------------
df = pd.DataFrame(results,
                  columns=["Accuracy","Inference Time","F1 Score"])

df["Model"] = model_list
df["TOPSIS Score"] = scores
df["Rank"] = ranks

df = df[["Model","Accuracy","Inference Time","F1 Score","TOPSIS Score","Rank"]]

print("\nFinal Results Table:\n")
print(df.round(4))

# --------------------------------------------------
# Visualization
# --------------------------------------------------

# Table Plot
plt.figure(figsize=(10,4))
plt.axis('off')

table = plt.table(
    cellText=df.round(4).values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)

table.scale(1,1.5)
table.auto_set_font_size(False)
table.set_fontsize(9)

plt.title("Model Comparison Table (TOPSIS - Text Classification)", pad=20)
plt.tight_layout()
plt.show()

# TOPSIS Bar Chart
plt.figure()
plt.bar(df["Model"], df["TOPSIS Score"])
plt.title("TOPSIS Ranking - Classification Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Accuracy Comparison
plt.figure()
plt.bar(df["Model"], df["Accuracy"])
plt.title("Accuracy Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
