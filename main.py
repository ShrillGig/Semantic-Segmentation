"""
This is the main file.
This is where all the image preparation for neural network training takes place.
From loading the dataset, to converting it into a neural network-friendly format.
In this file we compile the neural network, set up the hyperparameters for training and run it.
Don't forget to check the names of your neural networks in the separate file and in the main file, as well as the number of classes.
"""


#Make sure to import the correct neural network: from "neural_networks_file_name" import "neural_network_name_in_the_file"
from linknet import linknet_model #Importing a neural network from another file

from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from tensorflow.keras import metrics #Imported metrics from Tensorflow


#The image size you want to change to (if necessary)
SIZE_X = 128
SIZE_Y = 128

#Number of classes
n_classes = 4

#############################################################################################################################################
#############################################################################################################################################

#Read images and masks
BASE_PATH = Path("/content/drive/MyDrive/Colab Notebooks/") #Change to the route to your images and masks folder
IMAGE_DIR = BASE_PATH / "images-3" #Images folder
MASK_DIR = BASE_PATH / "masks-3"  #Masks folder

#Capture training image info as a list
train_images = []

for img_path in sorted(IMAGE_DIR.glob("*.jpg")): #Check your images format
    img = cv2.imread(str(img_path), 0) #Colored images will be read as grayscale
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    train_images.append(img) #Add image to the list

#Convert list to array for machine learning processing
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = []

for mask_path in sorted (MASK_DIR.glob("*.png")): #Check your masks format
    mask = cv2.imread(str(mask_path), 0) #Colored masks will be read as grayscale
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST) #Otherwise ground truth changes due to interpolation
    train_masks.append(mask) #Add mask to the list

#Convert list to array for machine learning processing
train_masks = np.array(train_masks)

print("x", train_masks.shape)
print("x", train_images.shape)

#############################################################################################################################################
#############################################################################################################################################

#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder() #Initialization
n, h, w = train_masks.shape #Get the shapes of masks (n-number of masks, h-height, w-width)
train_masks_reshaped = train_masks.reshape(-1,1) #Transform masks of size (n, h, w) into a two-dimensional array (n*h*w, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped.ravel()) #Find all unique classes and sort them in ascending order from 0
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w) #Return masks to original shape (n, h, w)

np.unique(train_masks_encoded_original_shape) #Output all extracted classes to the console for verification


#############################################################################################################################################
#############################################################################################################################################

train_images = np.expand_dims(train_images, axis=3) #Add a new axis to the array of images (n, h, w) and we get (n, h, w, 1). 1-if grascale images
train_images = train_images / 255.0 #Normalize pixels from the range [0, 255] to [0, 1]

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3) #Add a new axis to the array of masks (n, h, w) and we get (n, h, w, 1). 1-if grayscale masks are used

#############################################################################################################################################
#############################################################################################################################################

from sklearn.model_selection import train_test_split

# 1. Сначала разделяем на обучающую + тестовую и валидационную:
X_train_val, X_val, y_train_val, y_val = train_test_split(
    train_images, train_masks_input, test_size=215/2149, random_state=0)

# 2. Затем разделяем обучающую + тестовую на обучающую и тестовую:
X_train, X_test, y_train, y_test = train_test_split(
    X_train_val, y_train_val, test_size=194/1934, random_state=0)

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

#Convert training masks to One-hot-encoding format
from keras.utils import to_categorical
y_train_cat = to_categorical(y_train, num_classes=n_classes)
#y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

#Convert validation masks to One-hot-enoding format
y_val_cat = to_categorical(y_val, num_classes=n_classes)
#y_val_cat = val_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))

y_test_cat = to_categorical(y_test, num_classes=n_classes)
#y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

#From the X_train array retrieve image sizes
IMG_HEIGHT = X_train.shape[1]   #Heights
IMG_WIDTH  = X_train.shape[2]   #Width
IMG_CHANNELS = X_train.shape[3] #Channels

X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
X_test = X_test.astype("float32")

print("X_train dtype:", X_train.dtype)  # Должно быть float32
print("y_train_cat dtype:", y_train_cat.dtype)  # Должно быть float32 или float64

#############################################################################################################################################
#############################################################################################################################################

#Return the neural network we imported from another file and compile it
def get_model():
    return linknet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metrics.OneHotMeanIoU(n_classes)])
model.summary()

import tensorflow as tf
print(tf.__version__)
# Ensure TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Check for GPU devices
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")
    # Check your runtime type, install drivers etc.
    # if still GPU is unavailable, then comment out the below line
    #tf.config.experimental.set_visible_devices([], 'GPU') # Force using CPU if no GPU available


start_time = time.time()  #Start of the countdown
history = model.fit(X_train, y_train_cat,
                    batch_size = 16,
                    verbose=1,
                    epochs=100,
                    validation_data=(X_val, y_val_cat),
                    shuffle=False)

end_time = time.time() #End of the countdown

#Training time display
print(f"Trainig time: {end_time - start_time} seconds") #Get the time for which the neural network was trained (in seconds)

model.save("trained_model_linknet44.h5") #Save the trained model (if necessary)
model.summary()

#############################################################################################################################################
#############################################################################################################################################

#Plot the training and validation loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Plot the training and validation accuracy at each epoch
#Don't forget to change the last digit of the names acc and val_acc depending on what you see in the terminal

acc = history.history['one_hot_mean_io_u']
val_acc = history.history['val_one_hot_mean_io_u']

plt.plot(epochs, acc, 'y', label='Training MeanIoU')
plt.plot(epochs, val_acc, 'r', label='Validation MeanIOU')
plt.title('Training and validation MeanIoU')
plt.xlabel('Epochs')
plt.ylabel('MeanIoU')
plt.legend()
plt.show()

print(history.history.keys()) #Displays the available metrics and loss functions that were tracked during training


#############################################################################################################################################
#############################################################################################################################################

from keras.metrics import MeanIoU

# 1. Получаем предсказания
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

# 2. Убираем One-Hot Encoding
y_test_argmax = np.argmax(y_test_cat, axis=-1)

# 3. Вычисляем количество классов
num_classes_test = len(np.unique(y_test_argmax))
print("Number of classes in test set:", num_classes_test)

# 4. Создаем метрику MeanIoU
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)

# 5. Выводим Mean IoU
print("Mean IoU =", IOU_keras.result().numpy())

# 6. Получаем confusion matrix (НОВАЯ ВЕРСИЯ!)
values = IOU_keras.total_cm.numpy()  # Вместо get_weights()
print(values)

# 7. Вычисляем IoU для каждого класса
class_iou = []
for i in range(n_classes):
    intersection = values[i, i]  # Пересечение
    union = np.sum(values[i, :]) + np.sum(values[:, i]) - intersection  # Объединение
    iou = intersection / union if union != 0 else 0
    class_iou.append(iou)
    print(f"IoU for class {i} is: {iou:.4f}")


f1_metric = metrics.F1Score(average='weighted')
# Reshape y_test_cat and y_pred to 2D
y_test_cat_reshaped = y_test_cat.reshape(-1, n_classes)  # Reshape to (batch_size * height * width, num_classes)
y_pred_reshaped = y_pred.reshape(-1, n_classes)  # Reshape to (batch_size * height * width, num_classes)

f1_metric.update_state(y_test_cat_reshaped, y_pred_reshaped)
f1_score = f1_metric.result().numpy()
print("F1-score:", f1_score)

#############################################################################################################################################
#############################################################################################################################################

import random

# Количество предсказаний
num_predictions = 10

for j in range(num_predictions):
    # Выбираем случайное изображение из тестового набора
    test_img_number = random.randint(0, len(X_test) -1)
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)

    # Получаем предсказание
    prediction = model.predict(test_img_input)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    # Визуализируем результаты
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Test image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(232)
    plt.title('Mask')
    plt.imshow(ground_truth[:, :, 0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction')
    plt.imshow(predicted_img, cmap='jet')
    plt.show()
