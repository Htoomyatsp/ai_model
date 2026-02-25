from __future__ import annotations

from keras import Model
from keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
    Concatenate,
)
from keras.optimizers import Adam


def build_lstm_cnn(input_shape: tuple[int, int], output_dim: int) -> Model:
    inputs = Input(shape=input_shape, name="sequence")

    x = Conv1D(64, kernel_size=3, padding="causal", activation="relu", name="conv1")(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = Conv1D(64, kernel_size=3, padding="causal", activation="relu", name="conv2")(x)
    x = MaxPooling1D(pool_size=2, name="pool1")(x)
    x = LSTM(128, return_sequences=True, name="lstm1")(x)
    x = Dropout(0.2, name="dropout1")(x)
    x = LSTM(64, return_sequences=True, name="lstm2")(x)
    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dense(64, activation="relu", name="dense1")(x)
    outputs = Dense(output_dim, name="forecast")(x)

    model = Model(inputs=inputs, outputs=outputs, name="lstm_cnn")
    model.compile(optimizer=Adam(learning_rate=3e-4, clipnorm=1.0), loss="mse", metrics=["mae"])
    return model


def build_baseline_lstm(input_shape: tuple[int, int], output_dim: int) -> Model:
    inputs = Input(shape=input_shape, name="sequence")
    x = LSTM(128, return_sequences=True, name="lstm1")(inputs)
    x = Dropout(0.2, name="dropout1")(x)
    x = LSTM(64, return_sequences=True, name="lstm2")(x)
    x = LSTM(32, name="lstm3")(x)
    x = Dense(64, activation="relu", name="dense1")(x)
    outputs = Dense(output_dim, name="forecast")(x)

    model = Model(inputs=inputs, outputs=outputs, name="baseline_lstm")
    model.compile(optimizer=Adam(learning_rate=3e-4, clipnorm=1.0), loss="mse", metrics=["mae"])
    return model


def build_bi_lstm(input_shape: tuple[int, int], output_dim: int) -> Model:
    inputs = Input(shape=input_shape, name="sequence")

    x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm1")(inputs)
    x = Dropout(0.25, name="dropout1")(x)
    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm2")(x)
    x = Dropout(0.2, name="dropout2")(x)
    x = LSTM(32, return_sequences=True, name="lstm_tail")(x)
    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dense(64, activation="relu", name="dense1")(x)
    outputs = Dense(output_dim, name="forecast")(x)

    model = Model(inputs=inputs, outputs=outputs, name="bi_lstm")
    model.compile(optimizer=Adam(learning_rate=3e-4, clipnorm=1.0), loss="mse", metrics=["mae"])
    return model


def build_multi_input_hybrid(
    lookback: int,
    climate_feature_count: int,
    weather_feature_count: int,
    output_dim: int,
) -> Model:
    climate_in = Input(shape=(lookback, climate_feature_count), name="climate_sequence")
    weather_in = Input(shape=(lookback, weather_feature_count), name="weather_sequence")

    c = LSTM(96, return_sequences=True, name="climate_lstm1")(climate_in)
    c = LSTM(48, name="climate_lstm2")(c)

    w = Conv1D(32, kernel_size=3, padding="causal", activation="relu", name="weather_conv1")(weather_in)
    w = LSTM(32, name="weather_lstm1")(w)

    x = Concatenate(name="fusion")([c, w])
    x = Dense(96, activation="relu", name="fusion_dense1")(x)
    x = Dropout(0.2, name="fusion_dropout")(x)
    outputs = Dense(output_dim, name="forecast")(x)

    model = Model(inputs=[climate_in, weather_in], outputs=outputs, name="multi_input_hybrid")
    model.compile(optimizer=Adam(learning_rate=3e-4, clipnorm=1.0), loss="mse", metrics=["mae"])
    return model


def build_model(
    architecture: str,
    input_shape: tuple[int, int],
    output_dim: int,
    *,
    climate_feature_count: int | None = None,
    weather_feature_count: int | None = None,
) -> Model:
    architecture = architecture.lower().strip()
    builders = {
        "baseline_lstm": build_baseline_lstm,
        "lstm_cnn": build_lstm_cnn,
        "bi_lstm": build_bi_lstm,
        "multi_input_hybrid": None,
    }
    if architecture not in builders:
        supported = ", ".join(sorted(builders))
        raise ValueError(f"Unsupported architecture '{architecture}'. Supported: {supported}")
    if architecture == "multi_input_hybrid":
        if climate_feature_count is None or weather_feature_count is None:
            raise ValueError("multi_input_hybrid requires climate_feature_count and weather_feature_count")
        return build_multi_input_hybrid(
            lookback=input_shape[0],
            climate_feature_count=climate_feature_count,
            weather_feature_count=weather_feature_count,
            output_dim=output_dim,
        )
    return builders[architecture](input_shape=input_shape, output_dim=output_dim)
