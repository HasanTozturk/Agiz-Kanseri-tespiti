import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

'''# Veri yolu ve boyutları
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
)'''

# Xception modeli
base_model = Xception(weights='imagenet', include_top=False)
#base_model.trainable=False
for layer in base_model.layers[:10]:
    layer.trainable=False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Modeli derleme
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)  # Learning rate 0.001 olarak değiştirildi
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

for layer in model.layers://katman sayisini değiştirmek
    print(layer.trainable)

# TensorBoard geri çağırımı oluşturma
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)
model.save('xception.h5')//modeli kaydetmek

from keras.models import load_model//hazır modeli yüklemek için kullanılan kütüphane

  
# Modeli eğitme
"""history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=20,
    callbacks=[tensorboard_callback]
)

# Accuracy ve F1-score metriklerinin hesaplanması
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_f1 = [f1_score(test_generator.classes, (model.predict(test_generator) > 0.5).astype("int32")) for _ in range(1, len(train_accuracy) + 1)]
val_f1 = [f1_score(test_generator.classes, (model.predict(test_generator) > 0.5).astype("int32")) for _ in range(1, len(val_accuracy) + 1)]

# Grafik oluşturma
plt.figure(figsize=(10, 5))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Train Accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

# F1-score grafiği
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_f1) + 1), train_f1, label='Train F1 Score')
plt.plot(range(1, len(val_f1) + 1), val_f1, label='Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('Train and Validation F1 Score')
plt.legend()

plt.tight_layout()
plt.show()"""
