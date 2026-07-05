"""Load the Keras-2 NN models on TensorFlow >= 2.16 (Keras 3).

The .keras files under Models/NN_Models were saved with Keras 2 and don't
deserialize under Keras 3, so we force the tf-keras backend via
TF_USE_LEGACY_KERAS (set before tensorflow is imported). load_legacy_model
falls back to rebuilding from config.json + model.weights.h5 when tf-keras
can't read the archive directly.
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def load_legacy_model(path):
    """Load a Keras-2 ``.keras`` model on a modern TensorFlow/Keras stack."""
    import tf_keras

    path = str(path)
    try:
        return tf_keras.models.load_model(path, compile=False)
    except Exception:
        return _rebuild_from_archive(path)


def _rebuild_from_archive(path):
    """Fallback: rebuild architecture from config and map weights by layer."""
    import h5py
    import tf_keras

    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
        tmp_dir = tempfile.mkdtemp()
        archive.extract("model.weights.h5", tmp_dir)
        weights_path = os.path.join(tmp_dir, "model.weights.h5")

    model = tf_keras.models.model_from_config(config)
    model.build(model.input_shape)

    with h5py.File(weights_path, "r") as handle:
        layer_store = handle["_layer_checkpoint_dependencies"]
        for layer in model.layers:
            if layer.name not in layer_store:
                continue
            variables = layer_store[layer.name]["vars"]
            weights = [variables[str(i)][()] for i in range(len(variables.keys()))]
            if weights:
                layer.set_weights(weights)
    return model
