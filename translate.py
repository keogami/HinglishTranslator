import sys
import tensorflow as tf

if len(sys.argv) < 2:
    print("usage: python translate.py <text>")
    exit(-1)


try:
    normalizer = tf.saved_model.load('normalizer')
    translator = tf.saved_model.load('translator')
except Exception as e:
    print(e)

text = sys.argv[1]

result = normalizer.tf_translate(
    tf.constant([text])
)

result = translator.tf_translate(
    tf.constant([
        result['text'][0].numpy().decode()
    ])
)

print(result['text'][0].numpy().decode())