import os
import numpy
import matplotlib.pyplot as plt

# define the result statistics dataset

resnet18_SGD_Stats_Dict =  {"Training_time":"365m 40s", "Best_val_Acc": "0.519650", "Test_Loss": "0.421615", "Test_Accu": "51%"}
resnet18_ADAM_Stats_Dict =  {"Training_time":"155m 30s", "Best_val_Acc": "0.460550", "Test_Loss": "0.497545", "Test_Accu": "45%"}
resnet18_ADAMcustom_Stats_Dict =  {"Training_time":"146m 3s", "Best_val_Acc": "0.496600", "Test_Loss": "0.484651", "Test_Accu": "49%"}



# Extracting the required values from dictionaries
def extract_values(stats_dict):
    # Extracting training time and converting it to hours
    time_parts = stats_dict["Training_time"].split('m')
    minutes = int(time_parts[0])
    seconds = int(time_parts[1].strip('s'))
    training_time_hours = minutes + seconds / 60

    # Extracting best validation accuracy and converting it to percentage
    best_val_acc_percentage = float(stats_dict["Best_val_Acc"]) * 100

    # Extracting test accuracy and removing the "%" sign
    test_accuracy = int(stats_dict["Test_Accu"].strip('%'))

    return training_time_hours, best_val_acc_percentage, test_accuracy

# Initializing the lists
optimizers = ['SGD', 'ADAM', 'ADAM (Custom)']
training_times = []
best_val_acc = []
test_acc = []

# Extracting data for each optimizer
for stats in [resnet18_SGD_Stats_Dict, resnet18_ADAM_Stats_Dict, resnet18_ADAMcustom_Stats_Dict]:
    training_time, val_acc, test_accuracy = extract_values(stats)
    training_times.append(training_time)
    best_val_acc.append(val_acc)
    test_acc.append(test_accuracy)

# Printing the extracted values
print("Optimizers:", optimizers)
print("Training Times (hours):", training_times)
print("Best Validation Accuracy (%):", best_val_acc)
print("Test Accuracy (%):", test_acc)

# Create figure and axis objects with a shared x-axis
fig, ax1 = plt.subplots()

# Bar plot for training times
color = 'tab:red'
ax1.set_xlabel('Optimizer')
ax1.set_ylabel('Training Time (hours)', color=color)
ax1.bar(optimizers, training_times, color=color, alpha=0.6, label='Training Time')
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second y-axis for the accuracies
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color)  # We already handled the x-label with ax1
line1, = ax2.plot(optimizers, best_val_acc, color='tab:orange', marker='o', label='Best Validation Accuracy')
line2, = ax2.plot(optimizers, test_acc, color='tab:green', marker='x', label='Test Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# Adding text labels for accuracies
for i, txt in enumerate(best_val_acc):
    ax2.annotate(f'{txt}%', (optimizers[i], best_val_acc[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(test_acc):
    ax2.annotate(f'{txt}%', (optimizers[i], test_acc[i]), textcoords="offset points", xytext=(0,-15), ha='center')

# Adding a legend to the plot
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend([line1, line2], [labels2[0], labels2[1]], loc='upper left')

# Show plot
plt.title('Optimizer Performance Comparison')
plt.tight_layout()  # Adjust layout to make room for the legend
plt.show()
