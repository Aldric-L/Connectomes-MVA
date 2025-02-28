import numpy as np
import tensorflow as tf
import time as time

def default_1dCNN_encoder_builder(instructions, input_dim, output_dim, features):
    encoder_layers = [tf.keras.layers.InputLayer(shape=(1, input_dim))]
    for config in instructions:
        encoder_layers.append(tf.keras.layers.Conv1D(config['filters'], kernel_size=1, strides=config['stride']))
        encoder_layers.append(Resnet1DBlock(config['filters'], 1))

    encoder_layers.extend([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_dim)  # Latent distribution parameters
    ])
    encoder = tf.keras.Sequential(encoder_layers)
    return encoder

def default_1dCNN_decoder_builder(instructions, input_dim, output_dim, features):
    decoder_layers = [tf.keras.layers.InputLayer(shape=(input_dim,))]
    decoder_layers.append(tf.keras.layers.Reshape(target_shape=(1, input_dim)))

    for config in reversed(instructions):
        decoder_layers.append(Resnet1DBlock(config['filters'], 1, type='decode'))
        decoder_layers.append(tf.keras.layers.Conv1DTranspose(config['filters'], kernel_size=1, strides=1))

    # Final output layer
    decoder_layers.append(tf.keras.layers.Conv1DTranspose(output_dim, kernel_size=1, strides=1))
    decoder_layers.append(tf.keras.layers.Reshape(target_shape=(1,output_dim)))
    decoder = tf.keras.Sequential(decoder_layers)
    return decoder

def default_1dCNN_builder(instructions, input_dim, latent_dim, features):
    if features != 1:
        raise Exception("Trying to build a 1dCNN with more than one feature")

    # Building the encoder
    encoder = default_1dCNN_encoder_builder(instructions, input_dim, 2*latent_dim, 1)

    # Building the decoder
    decoder = default_1dCNN_decoder_builder(instructions, latent_dim, input_dim, 1)
    return encoder, decoder


def default_2dCNN_encoder_builder(instructions, input_dim, output_dim, features):
    encoder_layers = [tf.keras.layers.InputLayer(input_shape=(features, input_dim)),
                        tf.keras.layers.Reshape(target_shape=(features, input_dim, 1))]
    
    i = 0
    for config in instructions:
        encoder_layers.append(tf.keras.layers.Conv2D(config['filters'], kernel_size=config['kernel'][0], strides=config['stride'][0], activation=config['activ'][0], kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)))
        encoder_layers.append(tf.keras.layers.GroupNormalization(groups=-1))
        encoder_layers.append(tf.keras.layers.Conv2D(config['filters'], kernel_size=config['kernel'][1], strides=config['stride'][1], activation=config['activ'][1], kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)))
        if i != len(instructions)-1:
            encoder_layers.append(tf.keras.layers.GroupNormalization(groups=-1))
            #encoder_layers.append(tf.keras.layers.LeakyReLU(0.4))
            i += 1

    encoder_layers.extend([tf.keras.layers.Flatten(),
                           tf.keras.layers.Dense(output_dim, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))])
    encoder = tf.keras.Sequential(encoder_layers)
    return encoder

def default_2dCNN_decoder_builder(instructions, input_dim, output_dim, features, hidden_shape, hidden_length):
    # Building the decoder
    decoder_layers = [tf.keras.layers.InputLayer(input_shape=(input_dim,)),
                      tf.keras.layers.Dense(hidden_length, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)),
                      tf.keras.layers.Reshape(hidden_shape)]

    i = 0
    for config in reversed(instructions):
        decoder_layers.append(tf.keras.layers.Conv2DTranspose(config['filters'], kernel_size=config['kernel'][1], strides=config['stride'][1], activation=config['activ'][1], kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)))
        decoder_layers.append(tf.keras.layers.BatchNormalization())
        decoder_layers.append(tf.keras.layers.Conv2DTranspose(config['filters'], kernel_size=config['kernel'][0], strides=config['stride'][0], activation=config['activ'][0], kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)))
        if i != len(instructions)-1:
            decoder_layers.append(tf.keras.layers.BatchNormalization())
            #decoder_layers.append(tf.keras.layers.LeakyReLU(0.4))
            i += 1

    # Final output layer
    decoder_layers.extend([tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=1, strides=1, activation='linear'),
                           tf.keras.layers.Reshape(target_shape=(features, output_dim))])
    decoder = tf.keras.Sequential(decoder_layers)

    return decoder

def default_2dCNN_builder(instructions, input_dim, latent_dim, features):
    # Building the encoder
    encoder = default_2dCNN_encoder_builder(instructions, input_dim, 2*latent_dim, features)

    tmpshape = encoder.layers[len(encoder.layers)-2].input.shape[1:]
    tmplen = encoder.layers[len(encoder.layers)-2].output.shape[1]

    # Building the decoder
    decoder = default_2dCNN_decoder_builder(instructions, latent_dim, input_dim, features, tmpshape, tmplen)
    return encoder, decoder


class Resnet1DBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters,type='encode'):
        super(Resnet1DBlock, self).__init__(name=type+"_db_"+str(filters)+"_"+str(kernel_size))
    
        if type=='encode':
            self.conv1a = tf.keras.layers.Conv1D(filters, kernel_size, 2,padding="same")
            self.conv1b = tf.keras.layers.Conv1D(filters, kernel_size, 1,padding="same")
            self.norm1a = tf.keras.layers.GroupNormalization(groups=-1)
            self.norm1b = tf.keras.layers.GroupNormalization(groups=-1)
        if type=='decode':
            self.conv1a = tf.keras.layers.Conv1DTranspose(filters, kernel_size, 1,padding="same")
            self.conv1b = tf.keras.layers.Conv1DTranspose(filters, kernel_size, 1,padding="same")
            self.norm1a = tf.keras.layers.BatchNormalization()
            self.norm1b = tf.keras.layers.BatchNormalization()
        else:
            return None

    def call(self, input_tensor):
        x = tf.nn.relu(input_tensor)
        x = self.conv1a(x)
        x = self.norm1a(x)
        x = tf.keras.layers.LeakyReLU(0.4)(x)

        x = self.conv1b(x)
        x = self.norm1b(x)
        x = tf.keras.layers.LeakyReLU(0.4)(x)

        x += input_tensor
        return tf.nn.relu(x)

 # Configuration for the encoder and decoder
conv_config_placeholder = [
            {'filters': 32, 'stride': 2},
            {'filters': 64, 'stride': 2},
            {'filters': 128, 'stride': 2},
            {'filters': 256, 'stride': 2},
]

class ConvVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, input_dim, features, conv_config=conv_config_placeholder, CNN_builder=None):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.features = features

        self.maxvar = None
        self.minvar = None
        self.maxval = None
        self.minval = None

        if CNN_builder is None:
            if features == 1:
                CNN_builder = default_1dCNN_builder
            else:
                CNN_builder = default_2dCNN_builder
        
        if CNN_builder != -1:
            self.encoder, self.decoder = CNN_builder(conv_config, input_dim, latent_dim, features)
        else:
            self.encoder, self.decoder = None, None

    def set_varClipping(self, clip_min: int, clip_max: int):
        self.maxvar = clip_max
        self.minvar = clip_min

    def set_valClipping(self, clip_min: int, clip_max: int):
        self.maxval = clip_max
        self.minval = clip_min

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(200, self.latent_dim))
        return self.decode(eps)
    
    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        if self.minvar is not None and self.maxvar is not None:
            logvar = tf.clip_by_value(logvar, self.minvar, self.maxvar)
        return mean, logvar
    
    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if self.minval is not None and self.maxval is not None:
            logits = tf.clip_by_value(logits, self.minval, self.maxval)
        return logits
    
def frequency_loss(x, x_hat):
    x_fft = tf.signal.fft(tf.cast(x, tf.complex64))
    x_hat_fft = tf.signal.fft(tf.cast(x_hat, tf.complex64))

    magnitude_diff = tf.abs(tf.abs(x_fft) - tf.abs(x_hat_fft))
    return tf.reduce_mean(magnitude_diff)

def default_compute_loss(model, x, rloss_function, beta=1, epsilon = 1e-5, disp=False):
  mean, logvar = model.encode(x)
  tf.debugging.check_numerics(mean, "Mean contains NaN")
  tf.debugging.check_numerics(logvar, "Logvar contains NaN")
  z = model.reparameterize(mean, logvar + epsilon)
  tf.debugging.check_numerics(z, "Latent variable contains NaN")
  x_logit = model.decode(z)
  tf.debugging.check_numerics(x_logit, "Reconstruction contains NaN")

  #reconstruction_loss = tf.reduce_mean(rloss_function(x_logit, x), axis=1)
  reconstruction_loss = rloss_function(x_logit, x)

  kl_loss = tf.reduce_mean(-0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar + epsilon)))
  total_loss = reconstruction_loss + beta * kl_loss

  # Debugging output
  if disp:
      print('Total Loss: {}, Reconstruction Loss: {}, KL Divergence: {}, Beta: {}'
              .format(tf.reduce_sum(total_loss), tf.reduce_sum(reconstruction_loss), tf.reduce_sum(kl_loss), beta))
    
  return total_loss

@tf.function
def default_train_step(model, x, optimizer, compute_loss_function, **kwargs):
  """Executes one training step and returns the loss.

    Args:
        model: The model to train.
        x: Input data for the training step.
        optimizer: The optimizer used to update the model parameters.
        compute_loss: A callable that computes the loss given the model and input.
        **kwargs: Additional keyword arguments for the `compute_loss` function.

    Returns:
        The computed loss for the current training step.
    """
  with tf.GradientTape() as tape:
    loss = compute_loss_function(model, x, **kwargs)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

no_op = lambda *args, **kwargs: None

def default_train_VAE(epochs, train_dataset, test_dataset, model: ConvVAE, optimizer, 
                       loss_function, rloss_function, callback_function=no_op, **kwargs):
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(loss_function(model, test_x, rloss_function=rloss_function, beta=1, epsilon=1e-5, disp=True))
    elbo = -loss.result()
    callback_function(model=model, epoch=0, **kwargs)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(0, elbo, 0))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        beta = min(max(1, 0.5 + epoch / (epochs*0.5)), 3.5)
        for train_x in train_dataset:
            default_train_step(model, train_x, optimizer, loss_function, rloss_function=rloss_function, beta=beta)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(loss_function(model, test_x, rloss_function=rloss_function, beta=beta, epsilon=1e-5, disp=True))
        elbo = -loss.result()
        callback_function(model=model, epoch=epoch, **kwargs)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))