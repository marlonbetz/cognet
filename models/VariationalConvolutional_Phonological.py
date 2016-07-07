
#########
#NETWORK
############
import numpy as np
from keras.layers import Input, Dense, Lambda,merge,Convolution2D,Reshape,MaxPooling2D,UpSampling2D
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras import backend as K, objectives
from keras.regularizers import l2



batch_size = len(wordVectors.reshape((-1,nDim_phono)))
batch_size = 28
original_dim_phono = nDim_phono
original_dim_meaning = n_meanings
latent_dim = 1500
intermediate_dim_phono = 128
intermediate_dim_meaning = 128
intermediate_dim_concat = 128
epsilon_std = 0.01
nb_epoch = 200
#l2_value = 0.01
l2_value = 0

input_phono = Input(batch_shape=(batch_size, original_dim_phono))
#input_phono_corrupted= GaussianNoise(sigma=0.1,name="GaussianNoise")(input_phono)
#encoder phono
input_phono_img = Reshape((1,maxLength,nDim_PhonemeVectors))(input_phono)
x_phono = Convolution2D(64,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))(input_phono_img)
x_phono = MaxPooling2D((2,2))(x_phono)
x_phono = Convolution2D(32,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))(x_phono)
x_phono = MaxPooling2D((2,2))(x_phono)
h_phono = Reshape((32*5*50,))(x_phono)


x_meaning = Input(batch_shape=(batch_size, original_dim_meaning))

h_meaning = Dense(intermediate_dim_meaning, activation='relu')(x_meaning)

h_concat = merge([h_phono,h_meaning],mode="concat")
h_concat = Dense(intermediate_dim_concat, activation='relu')(h_concat)

z_mean = Dense(latent_dim)(h_concat)
z_log_std = Dense(latent_dim)(h_concat)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_std])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])

# we instantiate these layers separately so as to reuse them later
y_merged_layer = Dense(32*5*50,activation="relu")
y_merged = y_merged_layer(z)
#phono decoding
y_phono_reshaper = Reshape((32,5,50))
y_phono_reshaped = y_phono_reshaper(y_merged)
conv1_decoder = Convolution2D(32,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))
y_phono1 = conv1_decoder(y_phono_reshaped)
upSampling1_decoder = UpSampling2D((2,2))
y_phono2 = upSampling1_decoder(y_phono1)
conv2_decoder = Convolution2D(64,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))
y_phono3 = conv2_decoder(y_phono2)
upSampling2_decoder = UpSampling2D((2,2))
y_phono4 = upSampling2_decoder(y_phono3)
conv3_decoder = Convolution2D(1,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))
y_phono5 = conv3_decoder(y_phono4)
y_phono5_reshaper =  Reshape((nDim_phono,),name="decoded_phono")
decoded_phono = y_phono5_reshaper(y_phono5)


decoder_h_meaning = Dense(intermediate_dim_meaning, activation='relu')
decoder_mean_meaning = Dense(original_dim_meaning, activation='softmax',name="decoder_mean_meaning")
h_decoded_meaning = decoder_h_meaning(z)
x_decoded_mean_meaning = decoder_mean_meaning(h_decoded_meaning)

def vae_loss(x_phono,decoded_phono):
    ent_loss_meaning = objectives.categorical_crossentropy(x_meaning, x_decoded_mean_meaning)
    mse_loss_phono = objectives.mse(x_phono, decoded_phono)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
    return (ent_loss_meaning +
             mse_loss_phono + kl_loss)

vae = Model([input_phono,x_meaning], [decoded_phono,x_decoded_mean_meaning])
vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.fit(x=[wordVectors.reshape((-1,nDim_phono)),
           meanings.reshape((-1,n_meanings))], y=[wordVectors.reshape((-1,nDim_phono)),
                                                                                       meanings.reshape((-1,n_meanings))
                                                                                       ],
      batch_size=batch_size, nb_epoch=nb_epoch)
encoder = Model( [input_phono,x_meaning], z_mean)
embeddings = encoder.predict(x=[wordVectors.reshape((-1,nDim_phono)),
           meanings.reshape((-1,n_meanings))],batch_size=batch_size)

# generator, from latent space to reconstructed inputs


generator_input = Input(shape=(latent_dim,),name="generator_input")
#y_merged_layer = Dense(32*5*5,activation="relu")
y_merged_generator = y_merged_layer(generator_input)
#phono decoding
y_phono_reshaped_generator = y_phono_reshaper(y_merged_generator)
y_phono1_generator = conv1_decoder(y_phono_reshaped_generator)
y_phono2_generator = upSampling1_decoder(y_phono1_generator)
y_phono3_generator = conv2_decoder(y_phono2_generator)
y_phono4_generator = upSampling2_decoder(y_phono3_generator)
y_phono5_generator = conv3_decoder(y_phono4_generator)
decoded_phono_generator = y_phono5_reshaper(y_phono5_generator)

generator_phono = Model(generator_input, decoded_phono_generator)

h_decoded_meaning_generator = decoder_h_meaning(generator_input)
x_decoded_mean_meaning_generator = decoder_mean_meaning(h_decoded_meaning_generator)
generator_meaning = Model(generator_input, x_decoded_mean_meaning_generator)