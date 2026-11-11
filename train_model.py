
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path
import tensorflow as tf

# Project paths
ROOT = Path(__file__).resolve().parent 

# Actual extracted dataset location
DATA_DIR = Path(
    r"C:\PV\data\Plant_leave_diseases_dataset_without_augmentation"
)

OUT_MODEL = ROOT / "plant_disease_model.keras"
OUT_LABELS = ROOT / "class_names.json"

# Laptop-friendly settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 123
EPOCHS = 10

tf.keras.utils.set_random_seed(SEED)

# Check dataset
if not DATA_DIR.is_dir():
    raise SystemExit(f"Dataset not found: {DATA_DIR}")

class_dirs = sorted(
    p for p in DATA_DIR.iterdir() if p.is_dir()
)

if len(class_dirs) < 2:
    raise SystemExit("Not enough class folders found.")

print("\nPLANT DISEASE DATASET")
print("---------------------")
print("Classes:", len(class_dirs))

for folder in class_dirs:
    count = sum(
        1 for f in folder.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"{folder.name}: {count} images")
    if count == 0:
        raise SystemExit(f"Empty class folder: {folder.name}")

# Load training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

class_names = train_ds.class_names

if len(class_names) != len(class_dirs):
    raise SystemExit("Class folder count mismatch.")

print("\nLoaded classes:", len(class_names))

# Memory-friendly input pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(
    500, seed=SEED
).prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
])

# MobileNetV2 transfer learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save best model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(OUT_MODEL),
        monitor="val_accuracy",
        save_best_only=True
    )
]

print("\nStarting PlantVillage model training...")
print("Number of classes:", len(class_names))
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save model and class labels
model.save(OUT_MODEL)

OUT_LABELS.write_text(
    json.dumps(class_names, indent=2),
    encoding="utf-8"
)

print("\nTRAINING COMPLETED")
print("------------------")
print("Best validation accuracy:",
      round(max(history.history["val_accuracy"]), 4))
print("Model:", OUT_MODEL)
print("Labels:", OUT_LABELS)
