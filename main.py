import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalizing the data (changing range from 0-255 to 0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# building the model
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu)
    tf.keras.layers.Dense(128, activation=tf.nn.relu)
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# training the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

print(model.summary())

# compute loss and accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print('Validation loss: {} \n Validation accuracy: {}'.format(val_loss, val_acc))

# Save and load model
model.save('digit.model')
# load
new_model = tf.keras.models.load_model('digit.model')
# predictions
predictions = new_model.predict([x_test])
#print(predictions)
print(np.argmax(predictions[0]))

plt.imshow(x_test[0], cmap=plt.cm.binary) # cmp - color map (convert to black and white)
plt.show()
