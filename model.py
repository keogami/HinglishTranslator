import tensorflow as tf
import codec 

class TrainTranslator(tf.keras.Model):
    def __init__(self, embedding_dim, units,
                input_text_processor,
                output_text_processor):
        super().__init__()

        self.encoder = codec.Encoder(
            input_text_processor.vocabulary_size(),
            embedding_dim, units
        )

        self.decoder = codec.Decoder(
            output_text_processor.vocabulary_size(),
            embedding_dim, units
        )

        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

    def train_step(self, inputs):
        return self._tf_train_step(inputs)

    def _preprocess(self, input_text, targ_text):
        input_tokens = self.input_text_processor(input_text)
        targ_tokens = self.output_text_processor(targ_text)

        input_mask = input_tokens != 0
        targ_mask = targ_tokens != 0

        return input_tokens, input_mask, targ_tokens, targ_mask

    def _train_step(self, inputs):
        input_text, targ_text = inputs

        (
            input_tokens, input_mask,
            targ_tokens, targ_mask
        ) = self._preprocess(input_text, targ_text)

        max_targ_len = tf.shape(targ_tokens)[1]

        with tf.GradientTape() as tape:
            enc_output, enc_state = self.encoder(input_tokens)
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_targ_len-1):
                new_tokens = targ_tokens[:, t:t+2]
                step_loss, dec_state = self._loop_step(
                    new_tokens, input_mask,
                    enc_output, dec_state
                )
                loss = loss + step_loss

            average_loss = loss / tf.reduce_sum(tf.cast(targ_mask, tf.float32))
        
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {
            'batch_loss': average_loss
        }

    def _loop_step(self, new_tokens, input_mask,
                    enc_output, dec_state):
        input_tokens, targ_tokens = new_tokens[:, 0:1], new_tokens[:, 1:2]

        decoder_input = codec.DecoderInput(
            new_tokens=input_tokens,
            enc_output=enc_output, mask=input_mask
        )

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

        y = targ_tokens
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                                    tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)

