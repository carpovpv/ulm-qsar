import os
import logging
import warnings 

from config import UlmConfig

# Configure CUDA environment variables BEFORE importing TensorFlow
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Suppress INFO & WARNING C++ logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  

# Silence harmless Keras 3 EinsumDense mask propagation warning inside MultiHeadAttention
warnings.filterwarnings(
    "ignore",
    message=r".*was passed an input with a mask attached to it.*",
    category=UserWarning,
)

def init_tensorflow(
    cfg : UlmConfig
):
    """Initialize TensorFlow GPU settings, mixed precision, and random seeds.

    Must be called at the very top of your entrypoint script before importing
    any model or data modules that depend on TensorFlow.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_ids)

    import tensorflow as tf
    from tensorflow.keras import layers, mixed_precision

    tf.get_logger().setLevel(logging.ERROR)

    # Configure GPU memory growth so TF doesn't pre-allocate 100% of VRAM
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and cfg.memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Warning: Could not set memory growth on {gpu}: {e}")

    # Enable mixed precision ("mixed_float16", "mixed_bfloat16", or "float32")
    if cfg.precision_policy:
        mixed_precision.set_global_policy(cfg.precision_policy)

    # Set Python, NumPy, and TensorFlow seeds in one call
    if cfg.seed is not None:
        tf.keras.utils.set_random_seed(cfg.seed)

    return tf, layers


