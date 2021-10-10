import json
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from loss import MaskedLoss

from model import TrainTranslator
from translator import Translator

def tf_lower_and_split_punct(text):
    # text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)

    text = tf.strings.regex_replace(text, '[^ a-z.?!]', '')
    text = tf.strings.regex_replace(text, '[.?!,~:;]', r' \0 ')

    text = tf.strings.strip(text)

    text = tf.strings.join(['[SOS]', text, '[EOS]'], separator=' ')
    return text


inp = []
targ = []

with open("./data/translate.json") as input:
    data = json.load(input)

    for each in data:
        inp.append(each['hinglish'])
        targ.append(each['english'])

BUFFER_SIZE = len(inp)
BATCH_SIZE = 64

# inp = inp[:120]
# targ = targ[:120]

dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

max_vocab_size = 10000
input_text_processor = preprocessing.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size
)
input_text_processor.adapt(inp)

output_text_processor = preprocessing.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size
)
output_text_processor.adapt(targ)

embedding_dim = 256
units = 1024

train_translator = TrainTranslator(
    embedding_dim, units,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)

train_translator.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss(),
)

class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, key):
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])

batch_loss = BatchLogs('batch_loss')

train_translator.fit(dataset, epochs=3, callbacks=[batch_loss])

plt.plot(batch_loss.logs)
plt.ylim([0, 3])
plt.xlabel('Batch #')
plt.ylabel('CE/Token')

plt.savefig('./batch-loss-translator.pdf')

translator = Translator(
    encoder=train_translator.encoder,
    decoder=train_translator.decoder,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)

example = tf.constant([
    'hum baat pe khelne walon mei se nahi hai',
    'theek hai'
])

result = translator.translate(input_text=example)

for tr in result['text']:
    print(tr.numpy().decode())
print()

translator.export('translator')