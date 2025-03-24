import json
import numpy as np
from lilgradx.dataset import Dataset
from lilgradx.tensor import Tensor
from lilgradx.ll.mlp import MLP

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


dataset = Dataset(file_path="penguins.csv", target_column="species", drop_columns=['island', 'sex'])
_, X_test, _, y_test = dataset.get_data()


with open('model_state.json', 'r') as f:
    model_state = json.load(f)
print("Model state loaded successfully from 'model_state.json'.")

nin = model_state["config"]["nin"]
nouts = model_state["config"]["nouts"]
mlp = MLP(nin=nin, nouts=nouts)


saved_params = model_state["params"]
for param, saved_value in zip(mlp.parameters(), saved_params):
    param.data = saved_value

correct = 0
total = len(X_test)
y_true = []
y_pred = []

for x_np, y in zip(X_test, y_test):
    x = [Tensor(xi) for xi in x_np]
    output_probs = mlp(x)
    predicted_class = np.argmax([p.data for p in output_probs])
    actual_class = int(y.item())
    if predicted_class == actual_class:
        correct += 1
    y_true.append(actual_class)
    y_pred.append(predicted_class)

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(set(y_true))))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred))
