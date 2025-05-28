import numpy as np
from main import preprocess_spectrum

def test_preprocess_spectrum():
    dummy_spectrum = np.random.randint(100, 1000, 2048)
    processed = preprocess_spectrum(dummy_spectrum)
    assert len(processed) == 2048
    assert np.min(processed) >= 0
    assert np.max(processed) <= 1
