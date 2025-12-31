import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import json

# Configuration
DATASET_DIR = os.path.abspath("DataSet/train")
CSV_PATH = os.path.abspath("DataSet/train.csv")
MODEL_SAVE_PATH = os.path.abspath("app/models/action_recognition_model.h5")
CLASS_INDICES_PATH = os.path.abspath("app/models/class_indices.json")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1  # Adjustable based on time constraints

def create_model(num_classes):
    # CNN Part: MobileNetV2 for spatial feature extraction
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Freeze base model initially

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs)
    
    # Shape of x is (Batch, 7, 7, 1280)
    # We want to use LSTM for "reasoning".
    # Typically LSTM expects (Batch, Timesteps, Features).
    # We can treat the 7x7 spatial grid as a sequence of 49 vectors of size 1280.
    
    x = Reshape((49, 1280))(x)
    
    # LSTM Part
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    
    # Classification Head
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    
    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=DATASET_DIR,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=DATASET_DIR,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Save class indices
    class_indices = train_generator.class_indices
    inverted_indices = {v: k for k, v in class_indices.items()}
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(inverted_indices, f)
    print(f"Class indices saved to {CLASS_INDICES_PATH}")
    
    # Build Model
    num_classes = len(class_indices)
    model = create_model(num_classes)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    
    # Train
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Training complete.")

if __name__ == "__main__":
    main()
