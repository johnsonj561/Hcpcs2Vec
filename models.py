import tensorflow as tf
Keras = tf.keras
make_sampling_table = Keras.preprocessing.sequence.make_sampling_table
skipgrams = Keras.preprocessing.sequence.skipgrams
Model = Keras.models.Model
Dense, Dot = Keras.layers.Dense, Keras.layers.dot
Embedding, Reshape, Input = Keras.layers.Embedding, Keras.layers.Reshape, Keras.layers.Input
CSVLogger = Keras.callbacks.CSVLogger
EarlyStopping = Keras.callbacks.EarlyStopping


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


def skipgram_callbacks(csv_output_file):
    return [
        CSVLogger(output_file),
        EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)
    ]
