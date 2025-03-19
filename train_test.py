import pickle
from lilgradx.dataset import Dataset
from lilgradx.ll.mlp import MLP
from lilgradx.losses.losses import nll_loss, mse_loss
from lilgradx.tensor import Value  # Import the Value class
from lilgradx.ll.optimizer import Adam
import numpy as np

import dill as pickle
# Load dataset
dataset = Dataset(file_path="penguins.csv", target_column="species", drop_columns=['island', 'sex'])
X_train, X_test, y_train, y_test = dataset.get_data()


mlp = MLP(nin=len(X_train[0]), nouts=[6, 3])

# Set hyperparameters
lr = 0.01
epochs = 200
loss_function = "cross_entropy"  # Choose between "cross_entropy" or "mse"
training_accuracies = []

adam_optimizer = Adam(mlp.parameters(), lr=0.001)
# Training loop
for epoch in range(epochs):
    total_loss = 0
    correct = 0  # Track correct predictions for training accuracy
    for x_np, target_idx in zip(X_train, y_train):  # Iterate using X_train and y_train
        #print(x_np)
        x = [Value(xi) for xi in x_np]  # Convert x to Value objects
        output_probs = mlp(x)  # Forward pass

        # Compute loss
        if loss_function == "cross_entropy":
            loss = nll_loss(output_probs, target_idx.item())  # Cross-Entropy Loss
        elif loss_function == "mse":
            loss = mse_loss(output_probs, [Value(target_idx)])  # MSE Loss
        else:
            raise ValueError("Invalid loss function selected.")

        loss.backward()  # Backward pass
        total_loss += loss.data  # Accumulate total loss

        adam_optimizer.step()
        # Track correct predictions
        predicted_idx = np.argmax([p.data for p in output_probs])  # Get the index of the predicted class
        if predicted_idx == target_idx:
            correct += 1  # Increment correct count if prediction matches target
        mlp.zero_grad()  # Reset gradients
    # Compute training accuracy for this epoch
    train_accuracy = correct / len(y_train) * 100   
    training_accuracies.append(train_accuracy)

    avg_loss = total_loss / len(X_train)  # Calculate average loss
    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

print(f"Avg Train Accuracy: {np.mean(training_accuracies):.2f}%")

# After training, save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(mlp, f)  # Save the trained model

print("Model saved successfully.")

# Evaluate the model
correct = 0
total = len(X_test)




from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Initialize lists to store true labels and predicted labels
y_true = []
y_pred = []

# Evaluate the model on the test set
for x, y in zip(X_test, y_test):
    out = mlp(x)
    predicted_class = out.index(max(out, key=lambda v: v.data))  # Get index of max probability
    actual_class = int(y.item())
    if predicted_class == actual_class:
        correct += 1
    y_true.append(actual_class)
    y_pred.append(predicted_class)
    
accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")
# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(set(y_true))))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred))
