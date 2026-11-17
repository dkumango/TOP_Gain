#!/usr/bin/env python3
"""
topg_seqatt_bridge.py
=====================
Runs Sequential Attention (Yasuda et al., ICLR 2023; official Google Research code in topg_vendor_seqatt/)
in the Python 3.12 environment topg_venv_py312 with TensorFlow 2.16 and Keras 2 (tf_keras). The official
code relies on Keras 2 variable tracking: under Keras 3 its attention weights are not trained.
Called by topg_new_baselines.select_seqatt.

Usage: python topg_seqatt_bridge.py <in.csv> <out.txt> <max_k> <seed>
  in.csv : standardised features V0..Vp-1 and a last column y (integer class labels)
  out.txt: one 0-based feature index per line, in the order in which Sequential Attention selected them

Configuration (A9): one hidden layer of 67 units (official default), batch size 128, Adam with learning rate
1e-3 and no decay, no batch normalisation, training length fixed at 2,000 steps; one feature selected per step.
"""
import os
import sys

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "topg_vendor_seqatt"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

if os.environ.get("TOPG_N_JOBS"):  # runtime study only (topg_runtime_benchmark.py); unset in the benchmark
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TOPG_N_JOBS"]))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TOPG_N_JOBS"]))
from topg_seqatt_mlp_sa import SequentialAttentionModel  # noqa: E402

CONFIG = dict(hidden=[67], batch_size=128, learning_rate=1e-3, total_steps=2000)


def main():
    inp, out, max_k, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    df = pd.read_csv(inp)
    X = df.drop(columns="y").values.astype(np.float32)
    y = df["y"].values.astype(np.int32)
    k = min(max_k, X.shape[1])
    # tf.keras.utils.set_random_seed() fails under tf_keras 2.16 + Python 3.12 (random.randint with a float), so
    # the three generators are seeded directly
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    bs = min(CONFIG["batch_size"], X.shape[0])
    steps_per_epoch = int(np.ceil(X.shape[0] / bs))
    epochs = int(np.ceil(CONFIG["total_steps"] / steps_per_epoch))
    model = SequentialAttentionModel(num_inputs=X.shape[1], num_inputs_to_select=k,
                                     num_train_steps=epochs * steps_per_epoch, layer_sequence=CONFIG["hidden"],
                                     is_classification=True, num_classes=int(y.max()) + 1,
                                     learning_rate=CONFIG["learning_rate"], decay_steps=10 ** 9, decay_rate=1.0,
                                     batch_norm=False)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=model.optimizer)
    order = []

    def record():
        for j in np.flatnonzero(model.seqatt.selected_features.numpy() > 0.5):
            if int(j) not in order:
                order.append(int(j))

    class Track(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            record()

    ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(y), seed=seed).batch(bs)
    model.fit(ds, epochs=epochs, verbose=0, callbacks=[Track()])
    record()
    if len(order) < k:  # pending last step: rank the rest by the final attention weights
        w = model.seqatt._attention_weights.numpy()
        order += [int(j) for j in np.argsort(-w, kind="stable") if int(j) not in order]
    with open(out, "w") as f:
        f.write("\n".join(str(i) for i in order[:k]))


if __name__ == "__main__":
    main()
