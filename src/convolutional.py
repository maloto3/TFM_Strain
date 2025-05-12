from tensorflow.keras.regularizers import l2

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

def cnn(input_size: list,learning_rate: float, activation: str, dropout: float, regularization: float, num_classes: int, num_layers: int, num_filters: list, filter_size: int, pool_size:int):
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    if len(num_filters) != num_layers:
        raise ValueError("Number of filters and number of layers should be the same.")
    
    layers = [
        keras.layers.Conv3D(24, (2,2,3), strides = 1, activation='relu', input_shape=(input_size[0],input_size[1],3,1)), #Convolucion que sustituye a la cuantica
        keras.layers.Reshape((input_size[0]-1, input_size[1]-1, 24))
        ]
    for i in range(num_layers):
        layers.append(keras.layers.Conv2D(num_filters[i], filter_size, strides=1, activation=activation, padding="same", kernel_regularizer=l2(regularization)))
        layers.append(keras.layers.BatchNormalization(momentum=0.98))
        layers.append(keras.layers.Dropout(dropout))
        layers.append(keras.layers.MaxPooling2D(pool_size=pool_size))
    layers.append(keras.layers.Flatten())
    layers.append(keras.layers.Dense(num_classes, activation="softmax"))

    model = keras.models.Sequential(layers)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def qcnn(input_size: list, learning_rate: float, activation: str, dropout: float, regularization: float, num_classes: int, num_layers: int, num_filters: list, filter_size: int, pool_size:int):
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    if len(num_filters) != num_layers:
        raise ValueError("Number of filters and number of layers should be the same.")
    layers = []
    for i in range(num_layers):
        if i == 0:
            layers.append(keras.layers.Conv2D(num_filters[i], filter_size, strides=1, activation=activation, padding="same", kernel_regularizer=l2(regularization), input_shape=(input_size[0]-1,input_size[1]-1,24)))
        else:
            layers.append(keras.layers.Conv2D(num_filters[i], filter_size, strides=1, activation=activation, padding="same", kernel_regularizer=l2(regularization)))
        layers.append(keras.layers.BatchNormalization(momentum=0.98))
        layers.append(keras.layers.Dropout(dropout))
        layers.append(keras.layers.MaxPooling2D(pool_size=pool_size))
    layers.append(keras.layers.Flatten())
    layers.append(keras.layers.Dense(num_classes, activation="softmax"))

    model = keras.models.Sequential(layers)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



