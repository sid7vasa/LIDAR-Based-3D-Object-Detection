import tensorflow as tf

def custom_sparse_categorical_crossentropy(labels, preds, class_weights):
  # Convert the class weights dictionary into a tensor
  class_weights = tf.constant(list(class_weights.values()))

  # One-hot encode the labels
  one_hot_labels = tf.one_hot(labels, depth=10)

  # Apply categorical cross-entropy between the label and prediction
  loss = tf.keras.losses.categorical_crossentropy(one_hot_labels, preds, class_weights)

  return loss
