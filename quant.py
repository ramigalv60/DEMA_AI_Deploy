import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('models/bestmodel.keras')
model2 = tf.keras.models.load_model('models/my_model.keras')

# Convert the model to TensorFlow Lite format with dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter2 = tf.lite.TFLiteConverter.from_keras_model(model2)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_model2 = converter2.convert()

# Save the quantized model
with open('models/bestmodel_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

with open('models/my_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model2)