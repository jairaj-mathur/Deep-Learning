import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.utils import load_img
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
import os
import keras as kr
import tensorflow as tf
random.seed(a=None, version=2)

set_verbosity(INFO)


def get_mean_std_per_batch(image_dir, df, H=320, W=320):
    sample_data = []
    for img in df.sample(100)["Image"].values:
        image_path = os.path.join(image_dir, img)
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data, axis=(0, 1, 2, 3))
    std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1)
    return mean, std

pp_input=kr.applications.densenet.preprocess_input
def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
#     mean, std = get_mean_std_per_batch(image_dir, df, H=H, W=W)
#     img_path = os.path.join(image_dir, img)
    x = load_img(img, target_size=(H, W))#BECAUSE WE HAVE THE NAMES DIRECTLY
    if preprocess:
#         x -= mean
#         x /= std
        x= pp_input(np.array(x))
        x= np.expand_dims(x, axis=0)
    return x


def grad_cam(img_array, model, last_conv_layer_name, preds, pred_index=None, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    grad_model = kr.Model(model.get_layer('densenet121').inputs,
                          [model.get_layer('densenet121').get_layer('relu').output, 
                           model.get_layer('densenet121').output])


    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = pp_input(heatmap)
    return heatmap.numpy()


def compute_gradcam(model, img, image_dir, df, labels, selected_labels,
                    layer_name='densenet121'):
    
    preprocessed_input = load_image(img, image_dir, df, H=224, W=224)
    preds = model.predict(preprocessed_input)

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(preprocessed_input, model, layer_name, preds, pred_index=None, H=224, W=224)

#             gradcam = grad_cam(model, preprocessed_input, i, layer_name, H=224, W=224)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, df, preprocess=False),
                       cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1


def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        gt = generator.labels[:, i]
        pred = predicted_vals[:, i]
        auc_roc = roc_auc_score(gt, pred)
        auc_roc_vals.append(auc_roc)
        fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
        plt.figure(1, figsize=(10, 10))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_rf, tpr_rf,
                 label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
    plt.show()
    return auc_roc_vals
