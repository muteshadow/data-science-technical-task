```python
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D

np.random.seed(42)
tf.random.set_seed(42)
```


```python
import urllib.request

urllib.request.urlretrieve(
    "https://www.dropbox.com/scl/fi/b3n1j4sgb1qtma5l79rw4/train_happy.h5?rlkey=cq84k6ay3sfitllnk1wu2tt3k&dl=1",
    "train_happy.h5"
)

urllib.request.urlretrieve(
    "https://www.dropbox.com/scl/fi/20z852u9fhd0rmxcmiaq9/test_happy.h5?rlkey=11simxwvxrkcn9dcgmucbdrla&dl=1",
    "test_happy.h5"
)

def load_dataset(train_path, test_path):
    train_dataset = h5py.File(train_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])
    return train_set_x_orig, train_set_y_orig.reshape((1, train_set_y_orig.shape[0])), test_set_x_orig, test_set_y_orig.reshape((1, test_set_y_orig.shape[0])), classes

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset("train_happy.h5", "test_happy.h5")

X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print(f"X_train shape: {X_train.shape} | Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape} | Y_test shape: {Y_test.shape}")
```

    X_train shape: (600, 64, 64, 3) | Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3) | Y_test shape: (150, 1)
    


```python
def HappyModel(input_shape):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')
    return model
```


```python
happyModel = HappyModel((64, 64, 3))

happyModel.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

happyModel.summary()
```

    WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "HappyModel"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ zero_padding2d (<span style="color: #0087ff; text-decoration-color: #0087ff">ZeroPadding2D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">70</span>, <span style="color: #00af00; text-decoration-color: #00af00">70</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │           <span style="color: #00af00; text-decoration-color: #00af00">4,736</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn0 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">128</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pool (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32768</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ fc (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │          <span style="color: #00af00; text-decoration-color: #00af00">32,769</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">37,633</span> (147.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">37,569</span> (146.75 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">64</span> (256.00 B)
</pre>




```python
history = happyModel.fit(
    x=X_train, 
    y=Y_train, 
    validation_data=(X_test, Y_test),
    epochs=20, 
    batch_size=100,
    verbose=1
)
```

    Epoch 1/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 467ms/step - accuracy: 0.5233 - loss: 2.3813 - val_accuracy: 0.5600 - val_loss: 0.8208
    Epoch 2/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 402ms/step - accuracy: 0.5833 - loss: 1.3755 - val_accuracy: 0.5800 - val_loss: 0.6048
    Epoch 3/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 393ms/step - accuracy: 0.7833 - loss: 0.4557 - val_accuracy: 0.5867 - val_loss: 0.5869
    Epoch 4/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 395ms/step - accuracy: 0.8400 - loss: 0.3781 - val_accuracy: 0.5467 - val_loss: 0.7382
    Epoch 5/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 412ms/step - accuracy: 0.9033 - loss: 0.2340 - val_accuracy: 0.5867 - val_loss: 0.6655
    Epoch 6/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 400ms/step - accuracy: 0.9517 - loss: 0.1591 - val_accuracy: 0.5533 - val_loss: 0.6137
    Epoch 7/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 394ms/step - accuracy: 0.9267 - loss: 0.1531 - val_accuracy: 0.6133 - val_loss: 0.6403
    Epoch 8/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 391ms/step - accuracy: 0.9567 - loss: 0.1217 - val_accuracy: 0.6467 - val_loss: 0.5650
    Epoch 9/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 400ms/step - accuracy: 0.9717 - loss: 0.1011 - val_accuracy: 0.6667 - val_loss: 0.5361
    Epoch 10/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 386ms/step - accuracy: 0.9733 - loss: 0.0939 - val_accuracy: 0.6933 - val_loss: 0.5255
    Epoch 11/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 398ms/step - accuracy: 0.9800 - loss: 0.0842 - val_accuracy: 0.7400 - val_loss: 0.5036
    Epoch 12/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 396ms/step - accuracy: 0.9850 - loss: 0.0730 - val_accuracy: 0.7733 - val_loss: 0.4927
    Epoch 13/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 426ms/step - accuracy: 0.9850 - loss: 0.0688 - val_accuracy: 0.7933 - val_loss: 0.4725
    Epoch 14/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 386ms/step - accuracy: 0.9833 - loss: 0.0637 - val_accuracy: 0.8800 - val_loss: 0.4549
    Epoch 15/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 405ms/step - accuracy: 0.9900 - loss: 0.0568 - val_accuracy: 0.8933 - val_loss: 0.4474
    Epoch 16/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 394ms/step - accuracy: 0.9917 - loss: 0.0508 - val_accuracy: 0.8933 - val_loss: 0.4460
    Epoch 17/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 398ms/step - accuracy: 0.9900 - loss: 0.0479 - val_accuracy: 0.9133 - val_loss: 0.4331
    Epoch 18/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 406ms/step - accuracy: 0.9917 - loss: 0.0438 - val_accuracy: 0.9133 - val_loss: 0.4196
    Epoch 19/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 396ms/step - accuracy: 0.9933 - loss: 0.0402 - val_accuracy: 0.9267 - val_loss: 0.4091
    Epoch 20/20
    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 397ms/step - accuracy: 0.9933 - loss: 0.0365 - val_accuracy: 0.9333 - val_loss: 0.4052
    


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='royalblue', lw=2)
plt.plot(history.history['val_loss'], label='Test (Val) Loss', color='orangered', lw=2)
plt.title('Progress: Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='royalblue', lw=2)
plt.plot(history.history['val_accuracy'], label='Test (Val) Accuracy', color='orangered', lw=2)
plt.title('Progress: Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](hometask17_files/hometask17_5_0.png)
    

