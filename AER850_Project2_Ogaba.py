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
print("\n\n-----------------Step 2: Neural Network Architecture Design -----------------\n\n")

#-------------------------------------
#Part A: Neural Network Model Design and Compilation 
#-------------------------------------

from tensorflow.keras import layers, models

#Inputting our shapes from Step 1
input_shape = INPUT_SHAPE      
num_classes = 3

#Our custom CNN Model definition
def build_model(input_shape=INPUT_SHAPE, num_classes=3):
    model = models.Sequential([

        # --- Convolutional Block 1 ---
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # --- Convolutional Block 2 ---
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # --- Convolutional Block 3 ---
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # --- Additional Max Pooling Layer to reduce spatial size further ---
        layers.MaxPooling2D((2, 2)),

        # --- Flatten + Dense Layers ---
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


#Compiling the initial model 
model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


#Training our model for a total of 25 epochs 
epochs = 5
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

#-------------------------------------
#Part B: Initial Baseline Model Training Visualization and Performance Evaluation
#-------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

#Accurary Plot for visualization purposes prior to hyperparameter tuning 
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

#Loss Plot for visualization purposes prior to hyperparameter tuning 
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


































