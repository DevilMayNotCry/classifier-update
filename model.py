import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFViTModel, ViTConfig, TFViTForImageClassification

configuration = ViTConfig(hidden_size = 128,
                          num_hidden_layers = 8,
                          num_attention_heads = 8,
                          intermediate_size = 128,
                          hidden_act = "gelu")

base_model = TFViTForImageClassification(configuration)

pixel_values = tf.keras.layers.Input(shape=(3,224,224), name='pixel_values', dtype='float32')

vit = base_model.vit(pixel_values)[0]
x = tf.keras.layers.Dense(64, activation='relu')(vit[:, 0, :])
classifier = tf.keras.layers.Dense(6, activation='softmax', name='outputs')(x)


class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(factor=(-0.5, 0.5)),
                layers.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation",
        )
        #self.pixel_values = tf.keras.layers.Input(shape=(3,224,224), name='pixel_values', dtype='float32')
        self.configuration = ViTConfig(hidden_size = 128,
                          num_hidden_layers = 8,
                          num_attention_heads = 8,
                          intermediate_size = 128,
                          hidden_act = "gelu")
        self.base_model = TFViTForImageClassification(self.configuration)
        self.dense_layer = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(6, activation='softmax', name='outputs')


    def call(self, inputs):
  
        #pixel_values = self.pixel_values(inputs)
        print(input)
        
        x = self.data_augmentation(inputs)
        
        vit = self.base_model.vit(x)[0]
        x = self.dense_layer(vit[:, 0, :])
        classifier = self.output_layer(x)

        return classifier

keras_model = CustomModel()
