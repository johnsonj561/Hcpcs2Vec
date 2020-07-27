import tensorflow as tf
Keras = tf.keras
make_sampling_table = Keras.preprocessing.sequence.make_sampling_table
skipgrams = Keras.preprocessing.sequence.skipgrams
Model = Keras.models.Model
Dense, Dot = Keras.layers.Dense, Keras.layers.dot
Embedding, Reshape, Input = Keras.layers.Embedding, Keras.layers.Reshape, Keras.layers.Input
CSVLogger = Keras.callbacks.CSVLogger
EarlyStopping = Keras.callbacks.EarlyStopping
Callback = Keras.callbacks.Callback


def skipgram_model(vocab_size, embedding_size):
    # input and embedding layers
    input_target = Input((1,))
    input_context = Input((1,))
    embedding = Embedding(vocab_size, embedding_size,
                          input_length=1, name='embedding')

    # embed target and context
    target = embedding(input_target)
    target = Reshape((embedding_size, 1))(target)

    context = embedding(input_context)
    context = Reshape((embedding_size, 1))(context)

    # estimate similarity between target and context with dot product
    dot_product = Dot([target, context], axes=1)
    dot_product = Reshape((1,))(dot_product)

    # binary sigmoid output
    output = Dense(1, activation='sigmoid')(dot_product)
    output = Keras.layers.Flatten()(output)
    model = Keras.Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


class EpochTimerCallback(Keras.callbacks.Callback):
    def __init__(self, output_file):
        self.output_file = output_file
        self.t0 = time.time()
        self.times = []

    def on_epoch_end(self, epoch, logs, generator=None):
        delta = time.time() - self.t0
        self.t0 = time.time()
        self.times.append(str(delta))
        print(f'Completed epoch {epoch} in {delta} s')

    def on_train_end(self, logs=None):
        with open(self.output_file, 'a') as out:
            out.write()


def skipgram_callbacks(output_file):
    csv_out = f'loss-{output_file}'
    timing_out = f'epoch_time-{output_file}'
    return [
        CSVLogger(csv_out),
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        EpochTimerCallback(timing_out)
    ]
