import numpy as np
import tensorflow as tf
import time as time

from core.ConvVAE import ConvVAE, conv_config_placeholder, default_1dCNN_encoder_builder, default_1dCNN_decoder_builder, default_2dCNN_encoder_builder, default_2dCNN_decoder_builder, no_op

def cond_1dCNN_builder(instructions, input_dim, latent_dim, features, num_classes):
    if features != 1:
        raise Exception("Trying to build a 1dCNN with more than one feature")

    # Building the encoder
    encoder = default_1dCNN_encoder_builder(instructions, input_dim+num_classes, 2*latent_dim, 1)

    # Building the decoder
    decoder = default_1dCNN_decoder_builder(instructions, latent_dim+num_classes, input_dim, 1)
    return encoder, decoder

def cond_2dCNN_builder(instructions, input_dim, latent_dim, features, num_classes):
    # Building the theoretic encoder
    th_encoder = default_2dCNN_encoder_builder(instructions, input_dim, 2*latent_dim, features)

    tmpshape = th_encoder.layers[len(th_encoder.layers)-2].input.shape[1:]
    tmplen = th_encoder.layers[len(th_encoder.layers)-2].output.shape[1]

    # Building the true encoder
    encoder = default_2dCNN_encoder_builder(instructions, num_classes+input_dim, 2*latent_dim, features)

    # Building the decoder
    decoder = default_2dCNN_decoder_builder(instructions, num_classes+latent_dim, input_dim, features, tmpshape, tmplen)
    return encoder, decoder


class CondCVAE(ConvVAE):
    """Conditional Convolutional Variational Autoencoder."""

    def __init__(self, latent_dim: int, input_dim: int, features: int, num_classes: int, conv_config=conv_config_placeholder, CNN_builder=None):
        super(CondCVAE, self).__init__(latent_dim, input_dim, features, conv_config, -1)
        self.num_classes = num_classes

        if CNN_builder is None:
            if features == 1:
                CNN_builder = cond_1dCNN_builder
            else:
                CNN_builder = cond_2dCNN_builder
        
        self.encoder, self.decoder = CNN_builder(conv_config, input_dim, latent_dim, features, num_classes)

    @tf.function
    def encode(self, x, y):
        """Encode input data and condition on label."""
        #y_one_hot = tf.one_hot(y, depth=self.num_classes)
        y_one_hot = y
        y_one_hot = tf.expand_dims(y_one_hot, axis=1)  # Match dimensions of x
        #x_cond = tf.concat([x, tf.cast(y_one_hot, tf.float32)], axis=-1)
        x_cond = tf.concat([x, y_one_hot], axis=-1)
        mean, logvar = tf.split(self.encoder(x_cond), num_or_size_splits=2, axis=1)
        if self.minvar is not None and self.maxvar is not None:
            logvar = tf.clip_by_value(logvar, self.minvar, self.maxvar)
        return mean, logvar

    @tf.function
    def decode(self, z, y, apply_sigmoid=False):
        """Decode latent variable and condition on label."""
        #y_one_hot = tf.one_hot(y, depth=self.num_classes)
        y_one_hot = y
        #z_cond = tf.concat([z, tf.cast(y_one_hot, tf.float32)], axis=-1)
        z_cond = tf.concat([z, y_one_hot], axis=-1)
        logits = self.decoder(z_cond)
        if self.minval is not None and self.maxval is not None:
            logits = tf.clip_by_value(logits, self.minval, self.maxval)
        return logits

    @tf.function
    def sample(self, y, eps=None):
        """Sample from the latent space conditioned on y."""
        if eps is None:
            eps = tf.random.normal(shape=(1, self.latent_dim))
        return self.decode(eps, y)

# Updated loss and training functions to include conditioning
def cond_compute_loss(model, x, y, rloss_function, beta=1, epsilon=1e-5, disp=False):
    mean, logvar = model.encode(x, y)
    tf.debugging.check_numerics(mean, "Mean contains NaN")
    tf.debugging.check_numerics(logvar, "Logvar contains NaN")

    z = model.reparameterize(mean, logvar + epsilon)
    tf.debugging.check_numerics(z, "Latent variable contains NaN")

    x_logit = model.decode(z, y)
    tf.debugging.check_numerics(x_logit, "Reconstruction contains NaN")

    reconstruction_loss = rloss_function(x_logit, x)

    kl_loss = tf.reduce_mean(-0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar + epsilon)))
    total_loss = reconstruction_loss + beta * kl_loss

    # Debugging output
    if disp:
        print('Total Loss: {}, Reconstruction Loss: {}, KL Divergence: {}, Beta: {}'
              .format(tf.reduce_sum(total_loss), tf.reduce_sum(reconstruction_loss), tf.reduce_sum(kl_loss), beta))

    return total_loss

@tf.function
def cond_train_step(model, x, y, optimizer, compute_loss_function, **kwargs):
    with tf.GradientTape() as tape:
        loss = compute_loss_function(model, x, y, **kwargs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Modify the training loop for conditional VAE
def cond_train_CVAE(epochs, train_dataset, test_dataset, model: CondCVAE, optimizer, 
                    loss_function, rloss_function, callback_function=no_op, **kwargs):
    loss = tf.keras.metrics.Mean()
    for test_x, test_y in test_dataset:
        loss(loss_function(model, test_x, test_y, rloss_function=rloss_function, beta=1, epsilon=1e-5, disp=True))
    elbo = -loss.result()
    callback_function(model=model, epoch=0, **kwargs)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(0, elbo, 0))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        beta = max(1, 0.5 + epoch / (epochs * 0.3))
        for train_x, train_y in train_dataset:
            cond_train_step(model, train_x, train_y, optimizer, loss_function, rloss_function=rloss_function, beta=beta)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x, test_y in test_dataset:
            loss(loss_function(model, test_x, test_y, rloss_function=rloss_function, beta=beta, epsilon=1e-5, disp=True))
        elbo = -loss.result()
        callback_function(model=model, epoch=epoch, **kwargs)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))
