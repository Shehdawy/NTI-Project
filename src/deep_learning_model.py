"""Optional small Keras comparison model.

This is intentionally optional: the deployed app never depends on it, and
the project is explicit (in the README and in the JSON this function
returns) about whether it actually ran. If TensorFlow isn't installed, this
returns a "skipped" status rather than pretending a model was trained.
"""

from pathlib import Path


def train_deep_learning(X_train, y_train, X_test, y_test, output_path: Path) -> dict:
    """Train a small feed-forward network when TensorFlow is installed.

    Returns a dict with either ``{"status": "skipped", "reason": ...}`` or
    ``{"status": "trained", "mae": ..., "rmse": ...}`` -- callers should
    check ``status`` rather than assuming metrics are present.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return {"status": "skipped", "reason": "TensorFlow is not installed. Install tensorflow-cpu to run the optional experiment."}

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    callback = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=80, batch_size=64, callbacks=[callback], verbose=0)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return {"status": "trained", "mae": float(mae), "rmse": float(loss ** 0.5)}