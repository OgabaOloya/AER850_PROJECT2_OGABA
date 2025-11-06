# ================================
#Ogaba Oloya
#AER 850 Project 2
#501097689

#------------------------------------------------------------------------------
#Step 1: Data Processing 

#Importing our relevant toolkits for usage in this Project
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Defining our input image shape (Capitalized variable names for readabilities sake to indicate constants
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 500, 500, 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

#Defining our data directories that will be used 
base_dir = os.path.dirname(os.path.abspath(__file__))
train_directory = os.path.join(base_dir, 'train')
validation_directory = os.path.join(base_dir, 'valid')
test_directory = os.path.join(base_dir, 'test')        

# Training the data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,        
    shear_range=0.2,          
    zoom_range=0.2,          
    horizontal_flip=True       
)

# Validating our image generator for rescaling purposes
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

#Creating and training our validation generators 
train_generator = train_datagen.flow_from_directory(
    directory=train_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical'  
)

validation_generator = validation_datagen.flow_from_directory(
    directory=validation_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical'
)

#Testing our generator for evaluation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory=test_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")

#------------------------------------------------------------------------------
#Step 2: Neural Network Architecture Design 
#------------------------------------------------------------------------------

print("\n\n-----------------Step 2: Neural Network Architecture Design -----------------\n\n")

from tensorflow.keras import layers, models

#-------------------------------------
#Part A: Neural Network Model Design and Compilation 
#-------------------------------------

#Inputting our shapes from Step 1
input_shape = (500, 500, 3)      
num_classes = 3

#Our custom CNN Model definition
model = models.Sequential()

# --- Convolutional Block 1 ---
model.add(layers.Input(shape=input_shape))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# --- Convolutional Block 2 ---
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# --- Convolutional Block 3 ---
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

#Adding an additional Max Pooling Layer to reduce spatial size further
model.add(layers.MaxPooling2D((2, 2)))

#Our Flatten and Dense Layers ---
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

#Compiling the model 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model successfully compiled.\n")
print("Model Summary:\n")
model.summary()

#------------------------------------------------------------------------------
#Step 3: Hyperparameter Analysis 
#------------------------------------------------------------------------------

print("\n\n-----------------Step 3: Hyperparameter Analysis -----------------\n\n")

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#In this step, our adaptive hyperparameter controls are implemented to improve training efficiency.
#I have furthermore included earlyStopping which halts training when validation loss no longer improves, preventing overfitting.
#Additionally, ReduceLROnPlateau dynamically reduces the learning rate when improvement stagnates.

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

#Training our model using the defined callbacks for efficient convergence
epochs = 30
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete.")

#------------------------------------------------------------------------------
#Step 4: Model Evaluation
#------------------------------------------------------------------------------

print("\n\n-----------------Step 4: Model Evaluation -----------------\n\n")

import matplotlib.pyplot as plt 

#Extracting accuracy and loss metrics from our model training history
acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])

#Plotting Training and Validation Accuracy Curves
plt.figure(figsize=(8, 5))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("Figure_2_accuracy.jpg", dpi=200, bbox_inches="tight", format='jpg')
plt.show()

#Plotting Training and Validation Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("Figure_2_loss.jpg", dpi=200, bbox_inches="tight", format='jpg')
plt.show()

#-------------------------------------
#Part B: Model Evaluation Summary 
#-------------------------------------

#Printing out the saved figures for reference
print("\nSaved files:")
print("Figure_2_accuracy.jpg")
print("Figure_2_loss.jpg\n")

#Printing final training and validation metrics for summary comparison
if acc and val_acc and loss and val_loss:
    print(f"Final Training Accuracy: {acc[-1]:.3f}")
    print(f"Final Validation Accuracy: {val_acc[-1]:.3f}")
    print(f"Final Training Loss: {loss[-1]:.3f}")
    print(f"Final Validation Loss: {val_loss[-1]:.3f}")

print("\nModel evaluation completed successfully.")

#------------------------------------------------------------------------------
#Step 5 Preparation: Save Model for Testing Purposes
#------------------------------------------------------------------------------

print("\n\n-----------------Step 5 Preparation: Save Model for Testing -----------------\n\n")

#Saving our trained CNN model for usage in Step 5 (Model Testing)
model.save("Project_2_CNN.keras")

print("Model saved successfully as 'Project_2_CNN.keras'.")
print("This saved model will be reloaded in Step 5 for testing and final predictions.\n")
























