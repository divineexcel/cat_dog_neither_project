import os
from pathlib import Path
import tensorflow as tf
import keras
from keras import layers, models
from PIL import Image

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6
RESUME_EPOCHS = 1
MODEL_PATH = "models/classifier.keras"
HARD_EXAMPLE = Path("data/train/neither/human_group_01.jpg")
HARD_EXAMPLE_COPIES = 200


def ensure_hard_example_oversamples():
    """Give the supplied crowd scene enough weight to correct its dog error."""
    if not HARD_EXAMPLE.is_file():
        return

    with Image.open(HARD_EXAMPLE) as image:
        sample = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
        for index in range(1, HARD_EXAMPLE_COPIES + 1):
            copy_path = HARD_EXAMPLE.parent / f"human_group_hard_{index:03d}.jpg"
            if not copy_path.exists():
                sample.save(copy_path, format="JPEG", quality=92)

def main():
    ensure_hard_example_oversamples()
    training_epochs = EPOCHS
    print("Loading training and validation sets...")
    train_ds = keras.utils.image_dataset_from_directory(
        "data/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    val_ds = keras.utils.image_dataset_from_directory(
        "data/val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    if os.path.exists(MODEL_PATH):
        # Continue from the local checkpoint so retraining does not need to
        # download ImageNet weights or discard what the current model learned.
        print(f"Resuming from {MODEL_PATH}...")
        training_epochs = RESUME_EPOCHS
        model = keras.models.load_model(MODEL_PATH)
        if model.output_shape[-1] != len(class_names):
            raise ValueError(
                f"Checkpoint has {model.output_shape[-1]} outputs, "
                f"but the dataset has {len(class_names)} classes."
            )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(224, 224, 3))
        x = augmentation(inputs)
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(class_names), activation="softmax")(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    print("\nStarting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_epochs
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nTraining complete! Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
