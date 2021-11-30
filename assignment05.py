#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces

X,y = fetch_olivetti_faces(return_X_y=True)


# In[2]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 0.99)
X_pca = pca.fit_transform(X)


# In[3]:


X_pca.shape


# In[4]:


from sklearn.model_selection import train_test_split
# Given the scarcity of the dataset, an 80-20 ratio for train/test will be considered.
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, shuffle=True,random_state=42,stratify=y)


# In[5]:


input_size = X_pca.shape[1]
    
def mkmodel(top_size, center_size, regularization_rate):
    # Encoder
    encoder_input = tf.keras.layers.Input(shape=(input_size,))
    layer = tf.keras.layers.Dense(top_size, activation='tanh')(encoder_input)
    mean = tf.keras.layers.Dense(center_size, activation='relu', 
                                 kernel_regularizer=tf.keras.regularizers.l2(l=regularization_rate))(layer)
    var = tf.keras.layers.Dense(center_size, activation='relu', 
                                kernel_regularizer=tf.keras.regularizers.l2(l=regularization_rate))(layer)
    encoder_model = tf.keras.models.Model([encoder_input], [mean, var])

    # Decoder
    decoder_input = tf.keras.layers.Input((center_size,))
    layer = tf.keras.layers.Dense(top_size, activation='relu', kernel_regularizer=
                                  tf.keras.regularizers.l2(l=regularization_rate))(decoder_input)
    layer = tf.keras.layers.Dense(input_size)(layer)
    decoder_model = tf.keras.models.Model(decoder_input, layer)

    # Reparameterization Trick
    mean, var = encoder_model(encoder_input)
    epsilon = tf.random.normal(shape=(tf.shape(var)[0],tf.shape(var)[1]))
    z = mean + tf.exp(var) * epsilon
    decoder_output = decoder_model(z)
    model = tf.keras.models.Model(encoder_input, decoder_output)
    
#     loss = K.abs(decoder_output - encoder_input)
#     model.add_loss(loss)
    
    return model, decoder_model, encoder_model


# In[6]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# With 8 samples available for each person for training, We set the n_splits=4 , so for each split, we'll
# have 6 records per person to train and 2 record to validate with.
skf = StratifiedKFold(n_splits=4,shuffle=True,random_state=42)

top_layers = [180,130]
center_layers = [90,65]
regularization_rates = [0.001,0.005]
learning_rates = [0.01,0.001]

best_params = {
    "loss": np.Inf
}

for top in top_layers:
    for center in center_layers:
        for r in regularization_rates:
            for l in learning_rates:
                model,decoder,encoder = mkmodel(top, center, r)
                model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(l))
                losses = []
                for fit_idx, val_idx in skf.split(X_train,y_train):
                    X_fit, X_val = X_train[fit_idx], X_train[val_idx]
                    history = model.fit(X_fit,X_fit,epochs=50,validation_data=(X_val, X_val))
                    losses.append(history.history['val_loss'])
                loss = np.mean(losses)
                if best_params['loss'] > loss:
                    best_params['loss'] = loss
                    best_params['model'] = model
                    best_params['learning_rate'] = l
                    best_params['regularization_rate'] = r
                    best_params['n_top_layer'] = top
                    best_params['n_center_layer'] = center


# In[7]:


loss = best_params['loss']
model = best_params['model']
learning_rate = best_params['learning_rate']
regularization_rate = best_params['regularization_rate']
n_top_layer = best_params['n_top_layer']
n_center_layer = best_params['n_center_layer']


# In[8]:


print('loss:', loss)
print('learning_rate:', learning_rate)
print('regularization_rate:', regularization_rate)
print('n_top_layer:', n_top_layer)
print('n_center_layer:', n_center_layer)


# In[9]:


model,decoder,encoder = mkmodel(top, center, r)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(l))
history = model.fit(X_train,X_train,epochs=100,validation_data=(X_test, X_test))


# In[10]:


loss = history.history['loss']
val_loss = history.history['val_loss']


# In[11]:


epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[12]:


faces_decoded = model.predict(X_test)
for i in range(len(X_test)):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,5))
    
    original = pca.inverse_transform([X_test[i]])[0].reshape(64,64)
    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original")
    
    reconstructed = pca.inverse_transform([faces_decoded[i]])[0].reshape(64,64)
    ax2.imshow(reconstructed, cmap="gray")
    ax2.set_title("Reconstructed")

    
    plt.show()


# In[ ]:





# In[ ]:




