```python
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
tf.random.set_seed(42)
```


```python
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    X_train = np.array(train_dataset["train_set_x"][:]) 
    Y_train = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    X_test = np.array(test_dataset["test_set_x"][:]) 
    Y_test = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:]) 
    
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    
    return X_train, Y_train, X_test, Y_test, classes

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

print(f"X_train raw shape: {X_train_orig.shape} Y_train raw shape: {Y_train_orig.shape}")
print(f"X_test raw shape: {X_test_orig.shape} Y_test raw shape: {Y_test_orig.shape}")
print(f"Número de clases: {len(classes)} (caracteres del 0 al 5)")
```

    X_train raw shape: (209, 64, 64, 3) Y_train raw shape: (209, 1)
    X_test raw shape: (50, 64, 64, 3) Y_test raw shape: (50, 1)
    Número de clases: 2 (caracteres del 0 al 5)
    


```python
X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0

num_classes = 6
Y_train = to_categorical(Y_train_orig, num_classes)
Y_test = to_categorical(Y_test_orig, num_classes)

print(f"X_train: {X_train.shape} Y_train: {Y_train.shape}")
```

    X_train: (209, 64, 64, 3) Y_train: (209, 6)
    


```python
def create_model(activation_func='relu', dropout_rate=0.3):
    model = Sequential([
        Input(shape=(64, 64, 3)),
        Flatten(),
        
        Dense(256, activation=activation_func),
        Dropout(dropout_rate),

        Dense(128, activation=activation_func),
        Dropout(dropout_rate),

        Dense(64, activation=activation_func),
        Dropout(dropout_rate),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model(activation_func='tanh', dropout_rate=0.25)
model.summary()
```

    WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12288</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │       <span style="color: #00af00; text-decoration-color: #00af00">3,145,984</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │          <span style="color: #00af00; text-decoration-color: #00af00">32,896</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">8,256</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">390</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,187,526</span> (12.16 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,187,526</span> (12.16 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=40,
    batch_size=32,
    verbose=1
)
```

    Epoch 1/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 110ms/step - accuracy: 0.4737 - loss: 1.0479 - val_accuracy: 0.3400 - val_loss: 0.8284
    Epoch 2/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.5981 - loss: 0.7741 - val_accuracy: 0.3400 - val_loss: 0.7872
    Epoch 3/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 60ms/step - accuracy: 0.6029 - loss: 0.7346 - val_accuracy: 0.3400 - val_loss: 0.8157
    Epoch 4/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.5694 - loss: 0.7514 - val_accuracy: 0.3400 - val_loss: 1.0030
    Epoch 5/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step - accuracy: 0.6077 - loss: 0.7344 - val_accuracy: 0.3400 - val_loss: 0.7904
    Epoch 6/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6172 - loss: 0.7007 - val_accuracy: 0.3400 - val_loss: 0.8614
    Epoch 7/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6220 - loss: 0.7106 - val_accuracy: 0.3400 - val_loss: 0.8869
    Epoch 8/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.6172 - loss: 0.7156 - val_accuracy: 0.3400 - val_loss: 0.8082
    Epoch 9/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step - accuracy: 0.6507 - loss: 0.6548 - val_accuracy: 0.3400 - val_loss: 0.9245
    Epoch 10/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6268 - loss: 0.6716 - val_accuracy: 0.3400 - val_loss: 0.9227
    Epoch 11/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.6220 - loss: 0.7048 - val_accuracy: 0.3400 - val_loss: 0.8832
    Epoch 12/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 77ms/step - accuracy: 0.6029 - loss: 0.6958 - val_accuracy: 0.3400 - val_loss: 1.0688
    Epoch 13/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6220 - loss: 0.7131 - val_accuracy: 0.3400 - val_loss: 0.8426
    Epoch 14/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.5742 - loss: 0.7439 - val_accuracy: 0.3400 - val_loss: 1.0767
    Epoch 15/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - accuracy: 0.6029 - loss: 0.6866 - val_accuracy: 0.3400 - val_loss: 0.8048
    Epoch 16/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step - accuracy: 0.5694 - loss: 0.7205 - val_accuracy: 0.3400 - val_loss: 0.9162
    Epoch 17/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.6364 - loss: 0.6987 - val_accuracy: 0.3400 - val_loss: 0.9086
    Epoch 18/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6172 - loss: 0.7015 - val_accuracy: 0.3400 - val_loss: 0.9232
    Epoch 19/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.5837 - loss: 0.7220 - val_accuracy: 0.3400 - val_loss: 0.8023
    Epoch 20/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6172 - loss: 0.6656 - val_accuracy: 0.3400 - val_loss: 1.0524
    Epoch 21/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - accuracy: 0.6077 - loss: 0.6925 - val_accuracy: 0.3400 - val_loss: 0.8409
    Epoch 22/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.5502 - loss: 0.7272 - val_accuracy: 0.3400 - val_loss: 0.9769
    Epoch 23/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6124 - loss: 0.7042 - val_accuracy: 0.3400 - val_loss: 0.8879
    Epoch 24/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.5598 - loss: 0.6970 - val_accuracy: 0.3400 - val_loss: 0.9041
    Epoch 25/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6507 - loss: 0.6701 - val_accuracy: 0.3400 - val_loss: 0.9553
    Epoch 26/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step - accuracy: 0.6411 - loss: 0.6711 - val_accuracy: 0.3400 - val_loss: 0.9058
    Epoch 27/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.6268 - loss: 0.6792 - val_accuracy: 0.3400 - val_loss: 0.8896
    Epoch 28/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - accuracy: 0.5981 - loss: 0.7026 - val_accuracy: 0.3400 - val_loss: 0.9088
    Epoch 29/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.6124 - loss: 0.6445 - val_accuracy: 0.3400 - val_loss: 0.8557
    Epoch 30/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - accuracy: 0.6124 - loss: 0.6867 - val_accuracy: 0.3400 - val_loss: 0.8272
    Epoch 31/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - accuracy: 0.6507 - loss: 0.6367 - val_accuracy: 0.3400 - val_loss: 0.8348
    Epoch 32/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 70ms/step - accuracy: 0.6077 - loss: 0.6602 - val_accuracy: 0.3400 - val_loss: 0.8620
    Epoch 33/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.6699 - loss: 0.6534 - val_accuracy: 0.3400 - val_loss: 0.8251
    Epoch 34/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.6316 - loss: 0.6454 - val_accuracy: 0.3400 - val_loss: 0.9870
    Epoch 35/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6077 - loss: 0.7220 - val_accuracy: 0.3400 - val_loss: 0.8049
    Epoch 36/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - accuracy: 0.5933 - loss: 0.6867 - val_accuracy: 0.3400 - val_loss: 0.8902
    Epoch 37/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 76ms/step - accuracy: 0.6077 - loss: 0.6969 - val_accuracy: 0.3400 - val_loss: 0.9138
    Epoch 38/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 67ms/step - accuracy: 0.6364 - loss: 0.6371 - val_accuracy: 0.3400 - val_loss: 0.8168
    Epoch 39/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - accuracy: 0.6268 - loss: 0.6741 - val_accuracy: 0.3400 - val_loss: 0.9274
    Epoch 40/40
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - accuracy: 0.5981 - loss: 0.7115 - val_accuracy: 0.3400 - val_loss: 0.8530
    


```python
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


    
![png](hometask16_files/hometask16_5_0.png)
    

