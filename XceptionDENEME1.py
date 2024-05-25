import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import TensorBoard

# Veri yolu ve boyutları
train_path = 'dataset/train'
val_path = 'dataset/val'
test_path = 'dataset/test'
img_height, img_width = 299, 299
batch_size = 32

# Veri artırma ve ölçeklendirme
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
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Xception modeli
base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Modeli derleme
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# TensorBoard geri çağırımı oluşturma
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)

# Modeli eğitme
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=10,
    callbacks=[tensorboard_callback]
)


# Her epoch'ta ilerleme
for epoch, logs in enumerate(history.history['accuracy'], 1):
    print(f"Epoch {epoch}: Train - {logs}, Validation - {history.history['val_accuracy'][epoch-1]}")

# Toplam yüklenen ve değerlendirilen fotoğraf sayısı
total_loaded_images = (history.epoch[-1] + 1) * train_generator.samples
print(f"Toplam yüklenen fotoğraf sayısı: {total_loaded_images}")

# Modeli değerlendirme
# Tahminler
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype("int32")  # Tahminleri eşik değere göre sınıflara ayır

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(test_generator.classes, y_pred))

# Sınıflandırma raporu
print("Classification Report:")
print(classification_report(test_generator.classes, y_pred))
