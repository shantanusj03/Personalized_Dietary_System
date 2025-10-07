import tensorflow as tf
from tensorflow import keras
import numpy as np
import operator

MODEL_PATH = "path/to/your/model.h5"   # or path/to/saved_model_dir

# Candidate functions we will try mapping 'TrueDivide' to
candidates = [
    ("tf.math.divide", tf.math.divide),
    ("tf.divide", getattr(tf, "divide", None)),
    ("np.true_divide", np.true_divide),
    ("operator.truediv", operator.truediv),
    ("python_div", lambda x, y: x / y),
]

last_err = None
for name, fn in candidates:
    if fn is None:
        continue
    try:
        print(f"Trying custom_objects mapping -> 'TrueDivide': {name}")
        model = keras.models.load_model(MODEL_PATH, custom_objects={"TrueDivide": fn}, compile=False)
        print("Model loaded successfully using mapping:", name)
        break
    except Exception as e:
        print("Failed with mapping", name, " error:", repr(e))
        last_err = e
else:
    raise RuntimeError(
        "Tried all candidate mappings for 'TrueDivide' and failed. "
        "See earlier printed errors. If you want, share the model file or the code that created it."
    )
