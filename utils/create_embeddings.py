from deepface import DeepFace
import tensorflow as tf
import numpy as np



tflite_model_path = 'models\quantized_model_float16.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def get_embedding(image):
    input_shape = input_details[0]['shape']
    input_image = tf.image.resize(image, (input_shape[1], input_shape[2]))
    input_image = np.expand_dims(input_image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    return output_tensor[0]
