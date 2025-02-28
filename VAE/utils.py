import numpy as np
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, adjusted_mutual_info_score, adjusted_rand_score, f1_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
from scipy.stats import mode

def generate_and_save_images(model, epoch, test_sample, freq, save):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(18, 15))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        wave = np.asarray(predictions[i])
        librosa.display.waveshow(wave[0], sr=freq)
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        wave = np.asarray(test_sample[i])
        librosa.display.waveshow(wave[0], sr=freq)

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('outputs/{}_{:04d}.png'.format(save, epoch))
    plt.savefig('outputs/{}_{:04d}.png'.format(save, epoch))
    plt.show()

def build_tf_dataset(train_np_list, test_np_list, batch_size,max_training_len=None, max_testing_len=None, verbose=True):
    if max_training_len is not None and max_training_len > len(train_np_list):
        indices = np.random.choice(len(train_np_list), max_training_len, replace=False)
        train_audio_random = [train_np_list[i] for i in indices]
    else:
        train_audio_random = train_np_list
    
    if verbose is True:
        print("Raw train dataset completed.")

    if max_testing_len is not None and max_testing_len > len(test_np_list):    
        indices = np.random.choice(len(test_np_list), max_testing_len, replace=False)
        test_audio_random = [test_np_list[i] for i in indices[:max_testing_len]]
    else:
        test_audio_random = test_np_list

    train_size = len(train_audio_random)
    test_size = len(test_audio_random)
    if verbose is True:
        print("Raw datasets prepared!")

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_audio_random))
        .shuffle(train_size)
        .batch(batch_size)
    #   .map(lambda x: tf.expand_dims(x, 1))
    )
    
    if verbose is True:
        print("Train dataset completed")

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_audio_random))
        .shuffle(test_size)
        .batch(batch_size)
    #    .map(lambda x: tf.expand_dims(x, 1))
    )
    
    if verbose is True:
        print("Test dataset completed")

    if len(train_np_list.shape) <= 2:
        if verbose is True:
            print("Reshaping (adding 1 dimension)")
        # Reshaping audio data in the dataset
        train_dataset = train_dataset.map(
            lambda train_x: (tf.expand_dims(tf.cast(train_x, tf.float32), 1))
        )

        test_dataset = test_dataset.map(
            lambda test_x: (tf.expand_dims(tf.cast(test_x, tf.float32), 1))
        )

    if verbose is True:
        print("Train and test datasets prepared!")

    return train_dataset, test_dataset, train_size, test_size


def build_labelled_tf_dataset(train_np_list, test_np_list, train_labels_np_list, test_labels_np_list, num_classes, batch_size,max_training_len=None, max_testing_len=None, verbose=True):
    # Convert to one-hot encoding
    train_labels_one_hot = tf.cast(tf.one_hot(train_labels_np_list, num_classes), tf.float32)
    test_labels_one_hot = tf.cast(tf.one_hot(test_labels_np_list, num_classes), tf.float32)
    
    if max_training_len is not None and max_training_len > len(train_np_list):
        indices = np.random.choice(len(train_np_list), max_training_len, replace=False)
        train_audio_random = [train_np_list[i] for i in indices]
        train_label_random = [train_labels_one_hot[i] for i in indices]
    else:
        train_audio_random = train_np_list
        train_label_random = train_labels_one_hot
    
    if verbose is True:
        print("Raw train dataset completed.")

    if max_testing_len is not None and max_testing_len > len(test_np_list):    
        indices = np.random.choice(len(test_np_list), max_testing_len, replace=False)
        test_audio_random = [test_np_list[i] for i in indices[:max_testing_len]]
        test_label_random = [test_labels_one_hot[i] for i in indices[:max_testing_len]]
    else:
        test_audio_random = test_np_list
        test_label_random = test_labels_one_hot

    train_size = len(train_audio_random)
    test_size = len(test_audio_random)
    if verbose is True:
        print("Raw datasets prepared!")

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_audio_random, train_label_random))
        .shuffle(train_size)
        .batch(batch_size)
    #   .map(lambda x: tf.expand_dims(x, 1))
    )
    
    if verbose is True:
        print("Train dataset completed")

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_audio_random, test_label_random))
        .shuffle(test_size)
        .batch(batch_size)
    #    .map(lambda x: tf.expand_dims(x, 1))
    )
    
    if verbose is True:
        print("Test dataset completed")

    if len(train_np_list.shape) <= 3:
        if verbose is True:
            print("Reshaping (adding 1 dimension)")
        # Reshaping audio data in the dataset
        train_dataset = train_dataset.map(
            #lambda train_x, train_y: (tf.expand_dims(tf.cast(train_x, tf.float32), 1), tf.cast(train_y, tf.int8))
            lambda train_x, train_y: (tf.expand_dims(tf.cast(train_x, tf.float32), 1), train_y)
        )

        test_dataset = test_dataset.map(
            #lambda test_x, test_y: (tf.expand_dims(tf.cast(test_x, tf.float32), 1), tf.cast(test_y, tf.int8))
            lambda test_x, test_y: (tf.expand_dims(tf.cast(test_x, tf.float32), 1), test_y)

        )

    if verbose is True:
        print("Train and test datasets prepared!")

    return train_dataset, test_dataset, train_size, test_size


@tf.function
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
         -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=raxis)

def logit(p):
    return np.log(p) - np.log(1 - p)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def assess_clustering(cluster_labels, true_labels, latent_vectors):
    # Use linear sum assignment (Hungarian algorithm) to match clusters with true labels
    # Create a confusion matrix to match clusters to true labels
    #cm = confusion_matrix(true_labels, cluster_labels)
    #row_ind, col_ind = linear_sum_assignment(-cm)  # Find optimal assignment
    #cluster_labels_aligned = np.array([col_ind[label] for label in cluster_labels])  # Align clusters

    # Step 2: Map each cluster label to the most common true label
    cluster_to_true_label = {}
    for cluster in np.unique(cluster_labels):
        # Get the true labels of all data points in this cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_true_labels = true_labels[cluster_indices]
        
        # Find the most common true label in this cluster (mode)
        most_common_label = mode(cluster_true_labels).mode
        cluster_to_true_label[cluster] = most_common_label

    # Step 3: Assign the predicted label based on the most common true label for each cluster
    predicted_labels = np.array([cluster_to_true_label[label] for label in cluster_labels])

    # Step 4: Calculate accuracy (percentage of correct classifications)
    acc_score = np.mean(predicted_labels == true_labels)

    # Calculate clustering performance metrics with aligned labels
    #acc_score = accuracy_score(true_labels, cluster_labels_aligned)
    ami_score = adjusted_mutual_info_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    silhouette = silhouette_score(latent_vectors, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')

    # Print the evaluation metrics
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Adjusted Mutual Information (AMI): {ami_score:.3f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Accuracy Score: {acc_score:.3f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision Score: {precision:.4f}")

    return silhouette, nmi, ami_score, ari, acc_score, f1, precision