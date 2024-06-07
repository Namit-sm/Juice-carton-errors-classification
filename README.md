This repository contains code for training a Convolutional Neural Network (CNN) to classify images into three categories: "stained", "pressed", and "flawless". The model is implemented using Keras and TensorFlow on Google Colab.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Testing on New Images](#testing-on-new-images)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset is stored in Google Drive and contains images categorized into three classes:
- `stained`
- `pressed`
- `flawless`

Each category contains images of varying sizes which are resized to 200x200 pixels for training the model.

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Mount Google Drive in Google Colab:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. Ensure the dataset is placed in the correct directory in your Google Drive: `drive/MyDrive/automation_dataset`.

4. Install necessary dependencies:

    ```bash
    pip install tensorflow keras sklearn matplotlib
    ```

## Model Architecture

The CNN model consists of multiple convolutional layers with ReLU activations, followed by max-pooling layers, batch normalization, and fully connected dense layers. The model is designed to classify images into one of three categories.

```python
model = Sequential()

model.add(Conv2D(4, (3, 3), input_shape=(200, 200, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
Training the Model
To train the model, use the following code in Google Colab:

python
Copy code
history = model.fit(
    train_generator,
    steps_per_epoch=10,
    validation_data=validation_generator,
    validation_steps=10,
    epochs=50
)
Data augmentation is performed using ImageDataGenerator to improve model generalization.

Evaluating the Model
Evaluate the model on the test data:

python
Copy code
scores = model.evaluate(test_generator)
print("Accuracy =", scores[1])
Plot training and validation loss and accuracy:

python
Copy code
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
Testing on New Images
To test the model on new images, use the following code:

python
Copy code
IMG_SIZE = 200
img_array = cv2.imread('path_to_image.jpg', cv2.COLOR_BGR2RGB)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
prediction = model.predict(np.reshape(new_array, (1, 200, 200, 3)))
predicted_class = np.argmax(prediction)
print(predicted_class)
Results
Include details about the accuracy and performance of your model, along with some example predictions on new images.

Contributing
