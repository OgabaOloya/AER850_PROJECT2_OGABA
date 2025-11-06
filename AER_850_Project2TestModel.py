#------------------------------------------------------------------------------
#Step 5 Preparation: Save Model for Testing Purposes
#------------------------------------------------------------------------------

print("\n\n-----------------Step 5: Model Testing -----------------\n\n")

#I have split this section into multiple parts for ease of reading, editing and organizing

#-------------------------------------
#Part A: Model Loading and Preparation 
#-------------------------------------

#Importing our relevant toolkits for usage in this step
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

model = keras.models.load_model("Project_2_CNN.keras")

#Suppressing TensorFlow warnings for clean output readability
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings(
    "ignore",
    message="Skipping variable loading for optimizer",
    category=UserWarning
)

#Base directory and data paths
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, "train")
test_dir  = os.path.join(base_dir, "test")

#Image specifications
IMG_SHAPE = (500, 500)
BATCH_SIZE = 32

#-------------------------------------
#Part B: Class Indices and Reference Setup 
#-------------------------------------

#Creating a simple data generator to rebuild class index mapping
ref_gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
    train_dir,
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

#Extracting class name mappings
class_map = ref_gen.class_indices
index_to_class = {v: k for k, v in class_map.items()}
ordered_classes = [index_to_class[i] for i in range(len(index_to_class))]

print("Class mapping ('class': index):", class_map, "\n")

#-------------------------------------
#Part C: Image Preprocessing and Model Prediction 
#-------------------------------------

#Helper function for image processing
def prepare_image(file_path):
    """Loads an image, converts it to an array, and normalizes for model input."""
    img_loaded = load_img(file_path, target_size=IMG_SHAPE)
    img_array = img_to_array(img_loaded) / 255.0
    return img_loaded, np.expand_dims(img_array, axis=0)

#Test images to be evaluated
test_images = [
    ("crack", "test_crack.jpg", "Figure_3_test_crack.jpg"),
    ("missing-head", "test_missinghead.jpg", "Figure_3_test_missinghead.jpg"),
    ("paint-off", "test_paintoff.jpg", "Figure_3_test_paintoff.jpg")
]

#Iterative prediction for each image
records = []
for label_true, filename, output_fig in test_images:
    path = os.path.join(test_dir, label_true, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test image not found: {path}")

    img_disp, x_array = prepare_image(path)
    predictions = model.predict(x_array, verbose=0)[0]
    label_pred = ordered_classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100.0

    #Store results for summary display
    records.append({
        "Image": filename,
        "True Label": label_true,
        "Predicted": label_pred,
        "Confidence (%)": round(confidence, 2),
        "Probabilities": predictions
    })

    #-------------------------------------
    #Part D: Visualization of Prediction Results 
    #-------------------------------------

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    fig.suptitle(
        f"True Label: {label_true} | Predicted: {label_pred} ({confidence:.1f}%)",
        fontsize=11,
        y=0.98
    )

    ax[0].imshow(img_disp)
    ax[0].axis("off")

    ax[1].barh(
        [cls.replace('-', ' ').title() for cls in ordered_classes],
        predictions * 100,
        color="red"
    )
    ax[1].set_xlabel("Confidence (%)")
    ax[1].set_xlim(0, 100)
    ax[1].grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

#-------------------------------------
#Part E: Prediction Summary and Final Output 
#-------------------------------------

results_df = pd.DataFrame(records)
print("\nModel Testing Summary:\n")
print(results_df[["Image", "True Label", "Predicted", "Confidence (%)"]].to_string(index=False))

print("\nSaved prediction figures:")
for _, _, output_fig in test_images:
    print(output_fig)

print("\nModel testing successfully completed.\n")
