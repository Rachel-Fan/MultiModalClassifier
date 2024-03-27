import pandas as pd
import re

# Example log_data (Replace this variable's content with your actual logs)
log_data = """
Epoch: [0][  1/625]    Time  1.314 ( 1.612)    Data  1.169 ( 1.148)    Loss 4.0304e+00 (4.4452e+00)    Acc@1   7.03 (  8.82)   Acc@5  33.59 ( 24.75)
... (the rest of your log here)
val Loss: 2.1594 Acc: 0.5156
"""

# Regular expression patterns to match the lines and extract metrics
epoch_pattern = r"Epoch: \[(\d+)/\d+\]"
train_loss_pattern = r"train Loss: ([\d.e+-]+)"
train_acc_pattern = r"Acc: ([\d.e+-]+)"
val_loss_pattern = r"val Loss: ([\d.e+-]+)"
val_acc_pattern = r"Acc: ([\d.e+-]+)"

# Lists to store the extracted values
epochs = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []

# Split the log data into lines for processing
lines = log_data.split('\n')

# Variables to temporarily hold values for each epoch
current_epoch = None
temp_train_loss = None
temp_train_acc = None

for line in lines:
    epoch_match = re.search(epoch_pattern, line)
    if epoch_match:
        # When a new epoch is encountered, reset temporary values
        if current_epoch is not None:
            # Append values from the previous epoch to the lists
            epochs.append(current_epoch)
            train_losses.append(temp_train_loss)
            train_accs.append(temp_train_acc)
            # Reset temporary values for the new epoch
            temp_train_loss = None
            temp_train_acc = None
        current_epoch = int(epoch_match.group(1))
    else:
        # For lines within the same epoch, update temporary values with latest matches
        train_loss_match = re.search(train_loss_pattern, line)
        train_acc_match = re.search(train_acc_pattern, line)
        val_loss_match = re.search(val_loss_pattern, line)
        val_acc_match = re.search(val_acc_pattern, line)
        if train_loss_match:
            temp_train_loss = float(train_loss_match.group(1))
        if train_acc_match:
            temp_train_acc = float(train_acc_match.group(1))
        if val_loss_match and val_acc_match:
            # When validation metrics are found, append all epoch metrics to lists
            val_losses.append(float(val_loss_match.group(1)))
            val_accs.append(float(val_acc_match.group(1)))

# Ensure the last epoch's data is also appended
if current_epoch is not None:
    epochs.append(current_epoch)
    train_losses.append(temp_train_loss)
    train_accs.append(temp_train_acc)

# Create a DataFrame from the extracted values
df = pd.DataFrame({
    'Epoch': epochs,
    'Training Loss': train_losses,
    'Training Accuracy': train_accs,
    'Validation Loss': val_losses,
    'Validation Accuracy': val_accs
})

# Display the DataFrame
print(df)
