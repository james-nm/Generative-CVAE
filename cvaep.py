from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import warnings

from IPython import display

'''
IMPORTANT NOTES:
  - PEP8 issues: uses 2-space indents instead of 4-space
  - For learning purposes, #why comments are left in for searchability
'''

# TODO: Add capability to adjust image dimensions
#IMAGE_DIMENSIONS = [44, 44, 1]
IMAGE_DIMENSIONS = [28, 28, 1]
MODEL_FILENAME = 'model_weights'

class CVAE_Manager():
  
  def __init__(self, train_images = None, test_images = None, latent_dim = None):
    
    # TODO: Find better way to check type of these arguments
    if type(train_images) == type(None) or type(test_images) == type(None):
      # Load model from previously saved configuration
      self.latent_dim = latent_dim
      self.load_model()
      return
    
    # Generate new model and weights
    
    train_images = train_images.reshape(
        train_images.shape[0],
        IMAGE_DIMENSIONS[0],
        IMAGE_DIMENSIONS[1],
        IMAGE_DIMENSIONS[2]).astype('float32')
    test_images = test_images.reshape(
        test_images.shape[0],
        IMAGE_DIMENSIONS[0],
        IMAGE_DIMENSIONS[1],
        IMAGE_DIMENSIONS[2]).astype('float32')
    
    plt.figure(figsize=(4,4))
    for i in range(16):
      plt.subplot(4, 4, i+1)
      plt.imshow(train_images[i, :, :, 0], cmap='gray')
      plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(4,4))
    for i in range(16):
      plt.subplot(4, 4, i+1)
      plt.imshow(train_images[i, :, :, 0], cmap='gray')
      plt.axis('off')
    plt.show()
   
    # TODO investigate BUF
    TRAIN_BUF = 6000
    BATCH_SIZE = 100
    TEST_BUF = 1000
    
    self.train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
                    .shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    self.test_dataset = tf.data.Dataset.from_tensor_slices(test_images)\
                    .shuffle(TEST_BUF).batch(BATCH_SIZE)
    
    self.optimizer = tf.keras.optimizers.Adam(1e-4)
    
  def log_normal_pdf(self, sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
        
  @tf.function
  def compute_loss(self, model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
  
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit,
        labels=x)
    logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    logpz = self.log_normal_pdf(z, 0., 0.)
    logqz_x = self.log_normal_pdf(z, mean, logvar)
    
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    # bayesian properties-- monte carlo estimate
  
  @tf.function
  def compute_apply_gradients(self, model, x, optimizer):
    with tf.GradientTape() as tape:  # lookup tf.GradientTape()
        loss = self.compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
             
  def generate_and_save_images(self, model, epoch, test_input):
    # DEPRECATED
    # Replaced by generate_images()
    warnings.warn("Deprecated", DeprecationWarning)
    
    predictions = model.sample(test_input)
    plt.figure(figsize=(4, 4))
  
    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')
  
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    
  def generate_images(self, model, epoch, test_input, save=False):
    
    predictions = model.sample(test_input)

    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')
  
    if(save):
      plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    plt.show()
    
  def train(self, epochs=100, latent_dim=10):
    epochs = epochs
    latent_dim = latent_dim
    num_examples_to_generate = 16
    
    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim],
        dtype=tf.dtypes.float32)
    
    self.model = CVAE(latent_dim)
    
    self.generate_images(self.model, 0, random_vector_for_generation)
    
    for epoch in range(1, epochs + 1):
      start_time = time.time()
      for train_x in self.train_dataset:
        self.compute_apply_gradients(self.model, train_x, self.optimizer)
      end_time = time.time()
          
      if epoch % 1 == 0: # why is this necessary? TODO test without
        loss = tf.keras.metrics.Mean()
        for test_x in self.test_dataset:
          loss(self.compute_loss(self.model, test_x))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch{}'.format(
                  epoch,
                  elbo,
                  end_time-start_time))
        
        self.generate_images(self.model,
                             epoch,
                             random_vector_for_generation)
  def generate_unique(self, qty=1):
    
    self.model.latent_dim
    random_vector_for_generation = tf.random.normal(
        shape=[qty, self.model.latent_dim],
        dtype=tf.dtypes.float32)
    unique = self.model.sample(random_vector_for_generation)
    return unique.numpy()
     
  def generate_from_seed(self, seed, view_output=False):
    # Must be ran following the training process
    # TODO: verify that the model is ready / latent variables set
    
    # TODO: verify seed dimensionality, dtype
    
    mean, logvar = self.model.encode(seed)
    z = self.model.reparameterize(mean, logvar)
    x_logit = self.model.sample(z)
    
    predictions = x_logit.numpy()
    
    if view_output:
      # Reshape seed for display
      seed = seed[0, :, :, 0] * 255
      seed = seed.astype('uint8')
      
      plt.figure(figsize=(2,2))
      plt.subplot(1, 2, 1)
      plt.imshow(seed, cmap='gray')
      plt.axis('off')
      
      plt.subplot(1, 2, 2)
      plt.imshow(predictions[0, :, :, 0], cmap='gray')
      plt.axis('off')
      plt.title('Input and Output of Auto-encoder')
      plt.show()
      
      plt.figure()
      plt.title('generated from seed')
      plt.imshow(predictions[0, :, :, 0], cmap='gray', )
      plt.colorbar()
      plt.show()
    
    return predictions
  
  def save_model(self):
    # Save weights
    self.model.save_weights(MODEL_FILENAME)
    
    # Save configuration
    with open('config', encoding='utf-8', mode='w') as file:
      file.write(f'latent_dim:{self.model.latent_dim}')
    
  def load_model(self):
    
    # Clear any existing model data
    # TODO: Transfer all references of latent_dim to the model.
    #       This will require reworking the model constructor. 
    self.latent_dim = None
    self.model = None
    tf.keras.backend.clear_session()
    
    with open('config', encoding='utf-8', mode='r') as file:
      for line in file:
        contents = line.split(':')
        if contents[0] == 'latent_dim':
          self.latent_dim = int(contents[1])
  
    if self.latent_dim == None:
      raise Exception('Could not configure latent_dim')
    
    self.model = CVAE(self.latent_dim)
    self.model.load_weights(MODEL_FILENAME)
    
      
#define the CVAE Model
class CVAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(
              IMAGE_DIMENSIONS[0],
              IMAGE_DIMENSIONS[1],
              1)),
          tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Conv2D(
              filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
          tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
          tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ]
    )
                
  @tf.function
  def sample(self, eps=None):
    # EPS are the latent values for generative network input.
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)
    
  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x),
                            num_or_size_splits=2,
                            axis=1)
    return mean, logvar #what is logvar? Latent logits vector?
    
  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean
    
  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
        
    return logits