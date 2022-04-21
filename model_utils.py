import tensorflow as tf

def vgg19_layers(layer_names):
  vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet')
  print(vgg19.summary())
  vgg19.trainable = False
  outputs = [vgg19.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg19.input], outputs)
  return model
