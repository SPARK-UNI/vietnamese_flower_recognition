import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Đường dẫn dataset
train_dir = "train"
valid_dir = "valid"
test_dir  = "test"

img_size = (64, 64)   
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode="grayscale",  
    batch_size=batch_size,
    class_mode="categorical"
)

valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_gen.class_indices)

model = models.Sequential([
    layers.Flatten(input_shape=(img_size[0], img_size[1], 1)),  
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

history = model.fit(
    train_gen,
    epochs=150,
    validation_data=valid_gen,
    callbacks=[early_stop, reduce_lr]
)

loss, acc = model.evaluate(test_gen)
print(f"Test accuracy: {acc:.4f}")

model.save("ann_model.h5")
