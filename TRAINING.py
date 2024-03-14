import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# Define training and testing data directory
data_directory = r'C:\Users\User\Downloads\INDV\FYP\CSP600\archive\Train Images 13440x32x32\3. NORMALIZED'
test_data_directory = r'C:\Users\User\Downloads\INDV\FYP\CSP600\archive\Test Images 3360x32x32\test'

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Set to False to keep the order of predictions for later evaluation
)
train_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(128, 128),  # Adjust the target size as needed
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(28, activation='softmax'))  # num_classes is the number of classes for Arabic characters

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=validation_generator
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss}%")

# Print training accuracy
train_accuracy = history.history['accuracy'][-1]
print(f"Final Training Accuracy: {train_accuracy * 100:.2f}%")

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy',
         color='blue', marker='*')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy',
         color='orange', marker='*')
plt.axhline(y=test_accuracy, color='red', linestyle='--',
            label=f'Test Accuracy: {test_accuracy * 100:.2f}%')
plt.axhline(y=test_accuracy, color='green', linestyle='--',
            label=f'Train Accuracy: {train_accuracy * 100:.2f}%')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='*')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='*')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()  # Adjust layout
plt.show()

# Save the entire model in Keras format
model.save('arabic_model_sgd5.keras')
print("Trained model has been saved to 'arabic_model_sgd5.keras'")

# Evaluate the model on the test data
y_true = test_generator.classes
y_pred_probabilities = model.predict(test_generator)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(12, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = [str(i) for i in range(28)]  # Change the range based number of classes of Arabic Character
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Display numbers in the boxes
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white'
                 if conf_matrix[i, j] > conf_matrix.max() / 2
                 else 'black')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))
