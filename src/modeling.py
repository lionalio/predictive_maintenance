from libs import *
from data_processing import *


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def create_model(input_shape, n_classes, head_size, num_heads, ff_dim,
    num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    
    outputs = Dense(n_classes, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def train_model(X_tr, y_tr, features, label, scaler_path, window):
    scaler = get_scaling(X_tr, features, scaler_path)
    X_train = data_transform(
        X_tr, features, label,
        scaler, window, is_label=False
    )
    y_train = data_transform(
        y_tr, features, label,
        scaler, window, is_label=True
    )
    y_train = y_train[:, 0].reshape((-1,1))

    print(X_train.shape)
    print(y_train.shape)
    print(np.unique(y_train))
    input_shape = X_train.shape[1:]
    n_classes = y_train.shape[1] #len(np.unique(y_train))
    
    model = create_model(
        input_shape,
        n_classes,
        head_size=64,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=2,
        mlp_units=[64],
        mlp_dropout=0.2,
        dropout=0.25,
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    print(model.summary())

    callbacks = [EarlyStopping( 
        monitor='val_loss', min_delta=5e-3, 
        patience=10, verbose=0, mode='min',
        restore_best_weights=True
        )
    ]
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
    )

    model.save('../models/transformer.keras')


if __name__ == '__main__':
    with open('../data/processed/X_train.pkl', 'rb') as f1:
        X_train = pkl.load(f1)
    with open('../data/processed/X_test.pkl', 'rb') as f2:
        X_test = pkl.load(f2)
    with open('../data/processed/y_train.pkl', 'rb') as f3:
        y_train = pkl.load(f3)
    with open('../data/processed/y_test.pkl', 'rb') as f4:
        y_test = pkl.load(f4)
    
    # take only failure:
    y_train = y_train[:, 0].reshape((-1,1))
    y_test = y_test[:, 0].reshape((-1,1))

    print(X_train.shape)
    print(y_train.shape)
    print(np.unique(y_train))
    input_shape = X_train.shape[1:]
    n_classes = y_train.shape[1] #len(np.unique(y_train))
    
    model = create_model(
        input_shape,
        n_classes,
        head_size=64,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=2,
        mlp_units=[64],
        mlp_dropout=0.2,
        dropout=0.25,
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    print(model.summary())

    callbacks = [EarlyStopping( 
        monitor='val_loss', min_delta=5e-3, 
        patience=3, verbose=0, mode='min',
        restore_best_weights=True
        )
    ]
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
    )

    model.save('../models/transformer.keras')
    print(model.evaluate(X_test, y_test, verbose=1))

