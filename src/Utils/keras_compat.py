"""One Keras backend for NN training/serving across TF 2.14 and Keras 3."""
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras


def load_model(path, compile=False):
    """Try our preferred backend, then native Keras for newer .keras archives."""
    try:
        return keras.models.load_model(str(path), compile=compile)
    except Exception:
        from tensorflow import keras as native_keras
        if native_keras is keras:
            raise
        return native_keras.models.load_model(str(path), compile=compile)
