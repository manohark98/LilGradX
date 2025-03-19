import json
import numpy as np
from lilgradx.dataset import Dataset
from lilgradx.ll.mlp import MLP
from lilgradx.losses.losses import nll_loss, mse_loss
from lilgradx.tensor import Value
from lilgradx.ll.optimizer import Adam

# Load dataset
dataset = Dataset(file_path="penguins.csv", target_column="species", drop_columns=['island', 'sex'])
X_train, X_test, y_train, y_test = dataset.get_data()

nin = len(X_train[0])
nouts = [6, 3]
mlp = MLP(nin=nin, nouts=nouts)


epochs = 20
loss_function = "cross_entropy"  
training_accuracies = []
adam_optimizer = Adam(mlp.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    for x_np, target_idx in zip(X_train, y_train):
 
        x = [Value(xi) for xi in x_np]
        output_probs = mlp(x)
        
 
        if loss_function == "cross_entropy":
            loss = nll_loss(output_probs, target_idx.item())
        elif loss_function == "mse":
            loss = mse_loss(output_probs, [Value(target_idx)])
        else:
            raise ValueError("Invalid loss function selected.")

        loss.backward()  
        total_loss += loss.data
        
        adam_optimizer.step()  
        

        predicted_idx = np.argmax([p.data for p in output_probs])
        if predicted_idx == target_idx:
            correct += 1

        mlp.zero_grad()  

    train_accuracy = correct / len(y_train) * 100
    training_accuracies.append(train_accuracy)
    avg_loss = total_loss / len(X_train)
    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

print(f"Avg Train Accuracy: {np.mean(training_accuracies):.2f}%")

# Saves model parameters 
model_state = {
    "config": {
        "nin": nin,
        "nouts": nouts
    },
    "params": [p.data for p in mlp.parameters()]
}

with open('model_state.json', 'w') as f:
    json.dump(model_state, f)
print("Model state saved successfully as 'model_state.json'.")
