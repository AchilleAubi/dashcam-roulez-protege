try:
    import tensorflow.lite as tflite
    from tensorflow.lite.experimental.microfrontend.python.ops import load_delegate  
    TFLITE_AVAILABLE = True
except Exception:
    TFLITE_AVAILABLE = False
    load_delegate = None

def load_tflite_interpreter(model_path, use_edgetpu=False):
    if not TFLITE_AVAILABLE:
        raise RuntimeError("TensorFlow Lite non disponible.")
    
    # Pour EdgeTPU plus tard mais pour l'instant on ne l'utilise pas:
    delegates = []
    interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=delegates or None)
    interpreter.allocate_tensors()
    return interpreter
