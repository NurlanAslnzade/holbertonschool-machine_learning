#!/usr/bin/env python3
'''
training the data
'''


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.01, decay_rate=1, save_best=False,
                filepath=None):
    """
    training model
    """
    callbacks = []
    if validation_data is not None and early_stopping:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    if validation_data is not None and learning_rate_decay:

        def scheduler(epoch):
            """
            scheduler
            """
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        ))

    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath))

    optimizer = K.optimizers.SGD(learning_rate=alpha)

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle,
        callbacks=callbacks
    )
    return history
