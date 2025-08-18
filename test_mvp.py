#!/usr/bin/env python3
"""
Quick MVP test script for Tonkatsu-OS.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all main modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        from tonkatsu_os.database import RamanSpectralDatabase
        print("✅ Database module imported successfully")
    except Exception as e:
        print(f"❌ Database import failed: {e}")
        return False
    
    try:
        from tonkatsu_os.preprocessing import AdvancedPreprocessor
        print("✅ Preprocessing module imported successfully")
    except Exception as e:
        print(f"❌ Preprocessing import failed: {e}")
        return False
    
    try:
        from tonkatsu_os.ml import EnsembleClassifier
        print("✅ ML module imported successfully")
    except Exception as e:
        print(f"❌ ML import failed: {e}")
        return False
    
    try:
        from tonkatsu_os.api.main import app
        print("✅ FastAPI app imported successfully")
    except Exception as e:
        print(f"❌ API import failed: {e}")
        return False
    
    return True

def test_database():
    """Test database functionality."""
    print("\n🗄️ Testing database functionality...")
    
    try:
        from tonkatsu_os.database import RamanSpectralDatabase
        import numpy as np
        
        # Create test database
        db = RamanSpectralDatabase("test_db.db")
        
        # Test database stats
        stats = db.get_database_stats()
        print(f"✅ Database initialized. Spectra: {stats['total_spectra']}")
        
        # Test adding a spectrum
        test_spectrum = np.random.random(1000)
        spectrum_id = db.add_spectrum(
            compound_name="test_compound",
            spectrum_data=test_spectrum,
            chemical_formula="C6H6"
        )
        print(f"✅ Added test spectrum with ID: {spectrum_id}")
        
        # Test retrieval
        retrieved = db.get_spectrum_by_id(spectrum_id)
        if retrieved:
            print("✅ Successfully retrieved spectrum")
        else:
            print("❌ Failed to retrieve spectrum")
            return False
        
        # Cleanup
        db.close()
        os.remove("test_db.db")
        print("✅ Database test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing functionality."""
    print("\n🔬 Testing preprocessing functionality...")
    
    try:
        from tonkatsu_os.preprocessing import AdvancedPreprocessor
        import numpy as np
        
        preprocessor = AdvancedPreprocessor()
        
        # Create test spectrum
        test_spectrum = np.random.random(1000) * 1000 + 100
        
        # Test preprocessing
        processed = preprocessor.preprocess(test_spectrum)
        print(f"✅ Preprocessed spectrum: {len(processed)} points")
        
        # Test peak detection
        peaks, intensities = preprocessor.detect_peaks(processed)
        print(f"✅ Detected {len(peaks)} peaks")
        
        # Test feature extraction
        features = preprocessor.spectral_features(processed)
        print(f"✅ Extracted {len(features)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_api():
    """Test API functionality."""
    print("\n🌐 Testing API functionality...")
    
    try:
        from tonkatsu_os.api.main import app
        print("✅ FastAPI app loads successfully")
        
        # Just test that routes are configured
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        if len(routes) > 10:
            print(f"✅ API has {len(routes)} configured routes")
        else:
            print(f"❌ Expected more routes, got {len(routes)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔬 Tonkatsu-OS MVP Test Suite")
    print("=" * 40)
    
    all_passed = True
    
    # Run tests
    if not test_imports():
        all_passed = False
    
    if not test_database():
        all_passed = False
    
    if not test_preprocessing():
        all_passed = False
    
    if not test_api():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! MVP is ready!")
        print("\nTo start the application:")
        print("1. Backend:  make dev-backend")
        print("2. Frontend: make dev-frontend")
        print("   Or both:  make dev")
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())