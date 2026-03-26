from __future__ import annotations

from keras import Model, regularizers
from keras.layers import (
    LSTM,
    Add,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    Concatenate,
)
from keras.optimizers import Adam

# Mild L2 on all weight matrices — the single most impactful regularizer for
# small-to-medium time-series datasets.
_L2 = 1e-4


def _reg():
    return regularizers.l2(_L2)


# ---------------------------------------------------------------------------
# baseline_lstm
# ---------------------------------------------------------------------------

def build_baseline_lstm(input_shape: tuple[int, int], output_dim: int) -> Model:
    """
    Compact two-layer LSTM.  Reduced from 128/64/32 units to 64/32 to cut
    parameter count and prevent overfitting on typical greenhouse dataset sizes.
    Uses the last hidden state (return_sequences=False on the final layer) rather
    than pooling — correct for 1-step-ahead forecasting.
    """
    inputs = Input(shape=input_shape, name="sequence")
    x = LSTM(64, return_sequences=True, kernel_regularizer=_reg(), name="lstm1")(inputs)
    x = Dropout(0.2, name="drop1")(x)
    x = LSTM(32, return_sequences=False, kernel_regularizer=_reg(), name="lstm2")(x)
    x = Dense(32, activation="relu", kernel_regularizer=_reg(), name="dense1")(x)
    outputs = Dense(output_dim, name="forecast")(x)
    model = Model(inputs=inputs, outputs=outputs, name="baseline_lstm")
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------------
# lstm_cnn
# ---------------------------------------------------------------------------

def build_lstm_cnn(input_shape: tuple[int, int], output_dim: int) -> Model:
    """
    Causal CNN for local feature extraction, then LSTM for temporal context.
    Removed MaxPooling (was discarding temporal resolution on an already short
    window) and reduced unit counts.
    """
    inputs = Input(shape=input_shape, name="sequence")
    x = Conv1D(
        32, kernel_size=3, padding="causal", activation="relu",
        kernel_regularizer=_reg(), name="conv1",
    )(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = Conv1D(
        32, kernel_size=3, padding="causal", activation="relu",
        kernel_regularizer=_reg(), name="conv2",
    )(x)
    x = LSTM(64, return_sequences=False, kernel_regularizer=_reg(), name="lstm1")(x)
    x = Dropout(0.2, name="drop1")(x)
    x = Dense(32, activation="relu", kernel_regularizer=_reg(), name="dense1")(x)
    outputs = Dense(output_dim, name="forecast")(x)
    model = Model(inputs=inputs, outputs=outputs, name="lstm_cnn")
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------------
# bi_lstm
# ---------------------------------------------------------------------------

def build_bi_lstm(input_shape: tuple[int, int], output_dim: int) -> Model:
    """
    Bidirectional LSTM.  Key fix: removed GlobalAveragePooling1D — for 1-step
    forecasting the most recent hidden state carries the most signal, and
    averaging over the time axis dilutes it.  Reduced from 128→64→32 to 64→32
    to prevent overfitting.
    """
    inputs = Input(shape=input_shape, name="sequence")
    x = Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=_reg()),
        name="bilstm1",
    )(inputs)
    x = Dropout(0.2, name="drop1")(x)
    x = Bidirectional(
        LSTM(32, return_sequences=False, kernel_regularizer=_reg()),
        name="bilstm2",
    )(x)
    x = Dense(32, activation="relu", kernel_regularizer=_reg(), name="dense1")(x)
    outputs = Dense(output_dim, name="forecast")(x)
    model = Model(inputs=inputs, outputs=outputs, name="bi_lstm")
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------------
# temporal_conv  (new)
# ---------------------------------------------------------------------------

def _dilated_residual_block(
    x,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    name_prefix: str,
):
    """
    Causal dilated conv with LayerNorm and a residual projection if channel
    width changes.  Effective receptive field doubles with each stacked block.
    """
    conv = Conv1D(
        filters, kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
        kernel_regularizer=_reg(),
        name=f"{name_prefix}_conv",
    )(x)
    conv = LayerNormalization(name=f"{name_prefix}_ln")(conv)

    # Align channel width for the residual add
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding="same", name=f"{name_prefix}_proj")(x)
    return Add(name=f"{name_prefix}_add")([x, conv])


def build_temporal_conv(input_shape: tuple[int, int], output_dim: int) -> Model:
    """
    TCN-style dilated causal convolutions with residual connections.
    Receptive field with dilation 1/2/4/8 and kernel 3:
      (3-1)*(1+2+4+8) = 30 steps — covers the full lookback window at 48 steps.
    Terminates with a small LSTM to read the final temporal state cleanly.
    Typically the fastest to train and often matches or beats stacked LSTMs on
    smooth time-series.
    """
    inputs = Input(shape=input_shape, name="sequence")
    x = _dilated_residual_block(inputs, 32, 3, 1, "b1")
    x = _dilated_residual_block(x,      32, 3, 2, "b2")
    x = _dilated_residual_block(x,      32, 3, 4, "b3")
    x = _dilated_residual_block(x,      32, 3, 8, "b4")
    # Small LSTM reads the final causal state
    x = LSTM(32, return_sequences=False, kernel_regularizer=_reg(), name="lstm_tail")(x)
    x = Dropout(0.2, name="drop1")(x)
    x = Dense(32, activation="relu", kernel_regularizer=_reg(), name="dense1")(x)
    outputs = Dense(output_dim, name="forecast")(x)
    model = Model(inputs=inputs, outputs=outputs, name="temporal_conv")
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------------
# multi_input_hybrid
# ---------------------------------------------------------------------------

def build_multi_input_hybrid(
    lookback: int,
    climate_feature_count: int,
    weather_feature_count: int,
    output_dim: int,
) -> Model:
    """
    Separate pathways for inside-climate and outside-weather signals.
    Reduced from 96/48 + 32/32 to 48/24 + 24/24 to match dataset scale.
    """
    climate_in = Input(shape=(lookback, climate_feature_count), name="climate_sequence")
    weather_in = Input(shape=(lookback, weather_feature_count), name="weather_sequence")

    c = LSTM(48, return_sequences=True, kernel_regularizer=_reg(), name="climate_lstm1")(climate_in)
    c = LSTM(24, return_sequences=False, kernel_regularizer=_reg(), name="climate_lstm2")(c)

    w = Conv1D(
        24, kernel_size=3, padding="causal", activation="relu",
        kernel_regularizer=_reg(), name="weather_conv",
    )(weather_in)
    w = LSTM(24, return_sequences=False, kernel_regularizer=_reg(), name="weather_lstm")(w)

    x = Concatenate(name="fusion")([c, w])
    x = Dense(64, activation="relu", kernel_regularizer=_reg(), name="dense1")(x)
    x = Dropout(0.2, name="drop1")(x)
    outputs = Dense(output_dim, name="forecast")(x)

    model = Model(inputs=[climate_in, weather_in], outputs=outputs, name="multi_input_hybrid")
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_model(
    architecture: str,
    input_shape: tuple[int, int],
    output_dim: int,
    *,
    climate_feature_count: int | None = None,
    weather_feature_count: int | None = None,
) -> Model:
    architecture = architecture.lower().strip()
    single_input_builders = {
        "baseline_lstm": build_baseline_lstm,
        "lstm_cnn": build_lstm_cnn,
        "bi_lstm": build_bi_lstm,
        "temporal_conv": build_temporal_conv,
    }
    supported = sorted({*single_input_builders, "multi_input_hybrid"})
    if architecture not in supported:
        raise ValueError(f"Unsupported architecture '{architecture}'. Supported: {', '.join(supported)}")
    if architecture == "multi_input_hybrid":
        if climate_feature_count is None or weather_feature_count is None:
            raise ValueError("multi_input_hybrid requires climate_feature_count and weather_feature_count")
        return build_multi_input_hybrid(
            lookback=input_shape[0],
            climate_feature_count=climate_feature_count,
            weather_feature_count=weather_feature_count,
            output_dim=output_dim,
        )
    return single_input_builders[architecture](input_shape=input_shape, output_dim=output_dim)
