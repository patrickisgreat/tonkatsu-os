"""
Comprehensive Testing Suite for Tonkatsu-OS Raman Analysis System

This module provides unit tests, integration tests, and validation tests
for all components of the molecular identification system.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import sqlite3
import logging

# Import modules to test
from database import RamanSpectralDatabase
from preprocessing import AdvancedPreprocessor, PeakMatcher
from ml_models import EnsembleClassifier, ConfidenceScorer
from data_loader import SyntheticDataGenerator, DataIntegrator

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


class TestAdvancedPreprocessor(unittest.TestCase):
    """Test cases for the AdvancedPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = AdvancedPreprocessor()
        self.test_spectrum = np.random.random(2048) * 1000 + 100
        # Add some synthetic peaks
        self.test_spectrum[500:510] += 500
        self.test_spectrum[1000:1010] += 300
        self.test_spectrum[1500:1510] += 200
    
    def test_smooth_spectrum(self):
        """Test spectrum smoothing."""
        smoothed = self.preprocessor.smooth_spectrum(self.test_spectrum)
        
        self.assertEqual(len(smoothed), len(self.test_spectrum))
        self.assertIsInstance(smoothed, np.ndarray)
        
        # Smoothed spectrum should have less variation
        original_variance = np.var(self.test_spectrum)
        smoothed_variance = np.var(smoothed)
        self.assertLess(smoothed_variance, original_variance)
    
    def test_baseline_correction(self):
        """Test baseline correction."""
        # Add a baseline to test spectrum
        baseline = np.linspace(100, 200, len(self.test_spectrum))
        spectrum_with_baseline = self.test_spectrum + baseline
        
        corrected = self.preprocessor.baseline_correction(spectrum_with_baseline)
        
        self.assertEqual(len(corrected), len(self.test_spectrum))
        self.assertIsInstance(corrected, np.ndarray)
        
        # Mean should be closer to zero after baseline correction
        self.assertLess(abs(np.mean(corrected)), abs(np.mean(spectrum_with_baseline)))
    
    def test_normalize_spectrum(self):
        """Test spectrum normalization methods."""
        # Test min-max normalization
        normalized = self.preprocessor.normalize_spectrum(self.test_spectrum, method='minmax')
        self.assertAlmostEqual(np.min(normalized), 0.0, places=10)
        self.assertAlmostEqual(np.max(normalized), 1.0, places=10)
        
        # Test standard normalization
        std_normalized = self.preprocessor.normalize_spectrum(self.test_spectrum, method='standard')
        self.assertAlmostEqual(np.mean(std_normalized), 0.0, places=10)
        self.assertAlmostEqual(np.std(std_normalized), 1.0, places=10)
        
        # Test L2 normalization
        l2_normalized = self.preprocessor.normalize_spectrum(self.test_spectrum, method='l2')
        self.assertAlmostEqual(np.linalg.norm(l2_normalized), 1.0, places=10)
    
    def test_remove_cosmic_rays(self):
        """Test cosmic ray removal."""
        # Add artificial cosmic rays
        spectrum_with_spikes = self.test_spectrum.copy()
        spectrum_with_spikes[100] += 2000  # Add spike
        spectrum_with_spikes[500] += 1500  # Add spike
        
        cleaned = self.preprocessor.remove_cosmic_rays(spectrum_with_spikes)
        
        self.assertEqual(len(cleaned), len(spectrum_with_spikes))
        # Spikes should be reduced
        self.assertLess(cleaned[100], spectrum_with_spikes[100])
        self.assertLess(cleaned[500], spectrum_with_spikes[500])
    
    def test_detect_peaks(self):
        """Test peak detection."""
        peaks, peak_intensities = self.preprocessor.detect_peaks(self.test_spectrum)
        
        self.assertIsInstance(peaks, np.ndarray)
        self.assertIsInstance(peak_intensities, np.ndarray)
        self.assertEqual(len(peaks), len(peak_intensities))
        
        # Should find some peaks
        self.assertGreater(len(peaks), 0)
        
        # Peak intensities should correspond to spectrum values at peak positions
        for i, peak_pos in enumerate(peaks):
            self.assertAlmostEqual(peak_intensities[i], self.test_spectrum[peak_pos], places=5)
    
    def test_spectral_features(self):
        """Test spectral feature extraction."""
        features = self.preprocessor.spectral_features(self.test_spectrum)
        
        self.assertIsInstance(features, dict)
        
        # Check required features
        required_features = [
            'mean_intensity', 'std_intensity', 'max_intensity', 'min_intensity',
            'num_peaks', 'spectral_centroid', 'spectral_spread'
        ]
        
        for feature in required_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float, np.integer, np.floating))
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        processed = self.preprocessor.preprocess(self.test_spectrum)
        
        self.assertEqual(len(processed), len(self.test_spectrum))
        self.assertIsInstance(processed, np.ndarray)
        
        # Should be normalized (0-1 range)
        self.assertGreaterEqual(np.min(processed), 0.0)
        self.assertLessEqual(np.max(processed), 1.0)


class TestPeakMatcher(unittest.TestCase):
    """Test cases for the PeakMatcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.peak_matcher = PeakMatcher(tolerance=5.0)
        
        # Add some reference peaks
        self.peak_matcher.add_reference_peaks('compound_a', [100, 200, 300, 400])
        self.peak_matcher.add_reference_peaks('compound_b', [150, 250, 350, 450])
    
    def test_add_reference_peaks(self):
        """Test adding reference peaks."""
        self.peak_matcher.add_reference_peaks('test_compound', [500, 600, 700])
        
        self.assertIn('test_compound', self.peak_matcher.reference_peaks)
        np.testing.assert_array_equal(
            self.peak_matcher.reference_peaks['test_compound'],
            np.array([500, 600, 700])
        )
    
    def test_match_peaks(self):
        """Test peak matching functionality."""
        observed_peaks = np.array([102, 198, 305, 600])  # Some matches, some don't
        reference_peaks = np.array([100, 200, 300, 400])
        
        result = self.peak_matcher.match_peaks(observed_peaks, reference_peaks)
        
        self.assertIsInstance(result, dict)
        self.assertIn('matches', result)
        self.assertIn('match_score', result)
        self.assertIn('unmatched_observed', result)
        self.assertIn('unmatched_reference', result)
        
        # Should find 3 matches (102~100, 198~200, 305~300)
        self.assertEqual(len(result['matches']), 3)
        self.assertGreater(result['match_score'], 0.5)
    
    def test_identify_compound(self):
        """Test compound identification."""
        observed_peaks = np.array([101, 199, 301])  # Close to compound_a
        
        candidates = self.peak_matcher.identify_compound(observed_peaks)
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
        # Best match should be compound_a
        best_match = candidates[0]
        self.assertEqual(best_match['compound'], 'compound_a')
        self.assertGreater(best_match['match_score'], 0.5)


class TestRamanSpectralDatabase(unittest.TestCase):
    """Test cases for the RamanSpectralDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.db = RamanSpectralDatabase(self.temp_db.name)
        self.test_spectrum = np.random.random(2048)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.db.close()
        os.unlink(self.temp_db.name)
    
    def test_add_spectrum(self):
        """Test adding spectrum to database."""
        spectrum_id = self.db.add_spectrum(
            compound_name="test_compound",
            spectrum_data=self.test_spectrum,
            chemical_formula="C6H6",
            cas_number="71-43-2"
        )
        
        self.assertIsInstance(spectrum_id, int)
        self.assertGreater(spectrum_id, 0)
    
    def test_get_spectrum_by_id(self):
        """Test retrieving spectrum by ID."""
        # First add a spectrum
        spectrum_id = self.db.add_spectrum(
            compound_name="benzene",
            spectrum_data=self.test_spectrum,
            chemical_formula="C6H6"
        )
        
        # Retrieve it
        retrieved = self.db.get_spectrum_by_id(spectrum_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['compound_name'], "benzene")
        self.assertEqual(retrieved['chemical_formula'], "C6H6")
        np.testing.assert_array_equal(retrieved['spectrum_data'], self.test_spectrum)
    
    def test_search_by_compound_name(self):
        """Test searching by compound name."""
        # Add test spectra
        self.db.add_spectrum("benzene", self.test_spectrum)
        self.db.add_spectrum("toluene", self.test_spectrum)
        self.db.add_spectrum("benzyl alcohol", self.test_spectrum)
        
        # Exact search
        results = self.db.search_by_compound_name("benzene", exact_match=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['compound_name'], "benzene")
        
        # Partial search
        results = self.db.search_by_compound_name("benz", exact_match=False)
        self.assertEqual(len(results), 2)  # benzene and benzyl alcohol
    
    def test_database_stats(self):
        """Test database statistics."""
        # Add some test data
        for i in range(5):
            self.db.add_spectrum(f"compound_{i}", self.test_spectrum)
        
        stats = self.db.get_database_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_spectra'], 5)
        self.assertEqual(stats['unique_compounds'], 5)
        self.assertIn('top_compounds', stats)
    
    def test_search_similar_spectra(self):
        """Test similarity search."""
        # Add some spectra
        for i in range(3):
            spectrum = np.random.random(2048)
            self.db.add_spectrum(f"compound_{i}", spectrum)
        
        # Search for similar spectra
        query_spectrum = np.random.random(2048)
        results = self.db.search_similar_spectra(query_spectrum, top_k=2)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        if results:
            # Check result structure
            result = results[0]
            self.assertIn('compound_name', result)
            self.assertIn('similarity_score', result)
            self.assertIsInstance(result['similarity_score'], float)


class TestEnsembleClassifier(unittest.TestCase):
    """Test cases for the EnsembleClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = EnsembleClassifier(use_pca=True, n_components=10)
        
        # Generate synthetic training data
        np.random.seed(42)
        self.X_train = np.random.random((100, 50))
        self.y_train = np.random.choice(['class_a', 'class_b', 'class_c'], 100)
    
    def test_preprocess_features(self):
        """Test feature preprocessing."""
        X_processed = self.classifier.preprocess_features(self.X_train, fit=True)
        
        self.assertEqual(X_processed.shape[0], self.X_train.shape[0])
        self.assertEqual(X_processed.shape[1], self.classifier.n_components)
        
        # Mean should be approximately 0 after scaling
        self.assertLess(abs(np.mean(X_processed)), 0.1)
    
    def test_train(self):
        """Test model training."""
        results = self.classifier.train(self.X_train, self.y_train, validation_split=0.2)
        
        self.assertIsInstance(results, dict)
        self.assertTrue(self.classifier.is_trained)
        
        # Check required result keys
        required_keys = [
            'rf_accuracy', 'svm_accuracy', 'nn_accuracy', 'ensemble_accuracy',
            'n_train_samples', 'n_val_samples', 'n_features', 'n_classes'
        ]
        
        for key in required_keys:
            self.assertIn(key, results)
    
    def test_predict(self):
        """Test prediction functionality."""
        # Train the model first
        self.classifier.train(self.X_train, self.y_train)
        
        # Make predictions
        X_test = np.random.random((5, 50))
        predictions = self.classifier.predict(X_test)
        
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 5)
        
        # Check prediction structure
        pred = predictions[0]
        self.assertIn('predicted_compound', pred)
        self.assertIn('confidence', pred)
        self.assertIn('uncertainty', pred)
        self.assertIn('model_agreement', pred)
        self.assertIn('top_predictions', pred)
        self.assertIn('individual_predictions', pred)
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        # Train and save model
        self.classifier.train(self.X_train, self.y_train)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.classifier.save_model(temp_path)
            
            # Create new classifier and load model
            new_classifier = EnsembleClassifier()
            new_classifier.load_model(temp_path)
            
            self.assertTrue(new_classifier.is_trained)
            np.testing.assert_array_equal(
                new_classifier.class_names, 
                self.classifier.class_names
            )
            
        finally:
            os.unlink(temp_path)


class TestConfidenceScorer(unittest.TestCase):
    """Test cases for the ConfidenceScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.confidence_scorer = ConfidenceScorer()
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        probabilities = np.array([0.6, 0.3, 0.1])
        peak_match_score = 0.8
        model_agreement = 0.9
        spectral_quality = 0.95
        
        result = self.confidence_scorer.calculate_confidence_score(
            probabilities, peak_match_score, model_agreement, spectral_quality
        )
        
        self.assertIsInstance(result, dict)
        
        # Check required keys
        required_keys = [
            'overall_confidence', 'confidence_components', 
            'risk_level', 'recommendation'
        ]
        
        for key in required_keys:
            self.assertIn(key, result)
        
        # Overall confidence should be between 0 and 1
        self.assertGreaterEqual(result['overall_confidence'], 0.0)
        self.assertLessEqual(result['overall_confidence'], 1.0)
        
        # Risk level should be one of the expected values
        self.assertIn(result['risk_level'], ['low', 'medium', 'high'])


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test cases for the SyntheticDataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SyntheticDataGenerator()
    
    def test_generate_synthetic_spectrum(self):
        """Test synthetic spectrum generation."""
        spectrum = self.generator.generate_synthetic_spectrum('water')
        
        self.assertIsInstance(spectrum, np.ndarray)
        self.assertEqual(len(spectrum), 2048)
        
        # Spectrum should be non-negative
        self.assertGreaterEqual(np.min(spectrum), 0.0)
        
        # Should have some variation (not all zeros)
        self.assertGreater(np.max(spectrum), 0.1)
    
    def test_generate_training_dataset(self):
        """Test training dataset generation."""
        dataset = self.generator.generate_training_dataset(n_samples_per_compound=3)
        
        self.assertIsInstance(dataset, list)
        
        # Should have 3 samples per compound
        expected_samples = len(self.generator.common_compounds) * 3
        self.assertEqual(len(dataset), expected_samples)
        
        # Check sample structure
        sample = dataset[0]
        required_keys = ['compound_name', 'chemical_formula', 'spectrum_data', 'source']
        for key in required_keys:
            self.assertIn(key, sample)
        
        self.assertEqual(sample['source'], 'synthetic')
        self.assertIsInstance(sample['spectrum_data'], np.ndarray)


class TestDataIntegrator(unittest.TestCase):
    """Test cases for the DataIntegrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.db = RamanSpectralDatabase(self.temp_db.name)
        self.integrator = DataIntegrator(self.db)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.db.close()
        os.unlink(self.temp_db.name)
    
    def test_integrate_spectra(self):
        """Test spectrum integration."""
        test_spectra = [
            {
                'compound_name': 'test_compound_1',
                'spectrum_data': np.random.random(2048),
                'chemical_formula': 'C6H6',
                'source': 'test'
            },
            {
                'compound_name': 'test_compound_2',
                'spectrum_data': np.random.random(2048),
                'chemical_formula': 'C7H8',
                'source': 'test'
            }
        ]
        
        results = self.integrator.integrate_spectra(test_spectra)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(results['total_processed'], 2)
        self.assertEqual(results['successful_integrations'], 2)
        self.assertEqual(results['errors'], 0)
        
        # Verify data was actually integrated
        stats = self.db.get_database_stats()
        self.assertEqual(stats['total_spectra'], 2)
    
    def test_generate_and_integrate_synthetic(self):
        """Test synthetic data generation and integration."""
        results = self.integrator.generate_and_integrate_synthetic(n_samples_per_compound=2)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(results['successful_integrations'], 0)
        self.assertEqual(results['errors'], 0)
        
        # Verify data was integrated
        stats = self.db.get_database_stats()
        self.assertGreater(stats['total_spectra'], 0)


class TestValidation(unittest.TestCase):
    """Integration tests and validation tests."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.db = RamanSpectralDatabase(self.temp_db.name)
        self.preprocessor = AdvancedPreprocessor()
        self.classifier = EnsembleClassifier(use_pca=True, n_components=20)
        self.integrator = DataIntegrator(self.db)
        
        # Generate and integrate test data
        self.integrator.generate_and_integrate_synthetic(n_samples_per_compound=10)
    
    def tearDown(self):
        """Clean up integration test environment."""
        self.db.close()
        os.unlink(self.temp_db.name)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Generate synthetic spectrum
        generator = SyntheticDataGenerator()
        test_spectrum = generator.generate_synthetic_spectrum('water')
        
        # 2. Preprocess spectrum
        processed = self.preprocessor.preprocess(test_spectrum)
        self.assertIsInstance(processed, np.ndarray)
        
        # 3. Extract features
        features = self.preprocessor.spectral_features(processed)
        self.assertIsInstance(features, dict)
        
        # 4. Search similar spectra
        similar = self.db.search_similar_spectra(test_spectrum, top_k=3)
        self.assertIsInstance(similar, list)
        
        # 5. Train classifier (if enough data)
        stats = self.db.get_database_stats()
        if stats['total_spectra'] >= 10:
            # This would require implementing data loading from database
            # For now, we'll use synthetic data
            X_train = np.random.random((50, 100))
            y_train = np.random.choice(['water', 'ethanol', 'benzene'], 50)
            
            results = self.classifier.train(X_train, y_train)
            self.assertIsInstance(results, dict)
            self.assertTrue(self.classifier.is_trained)
    
    def test_database_consistency(self):
        """Test database consistency and integrity."""
        stats = self.db.get_database_stats()
        
        # Verify database has data
        self.assertGreater(stats['total_spectra'], 0)
        self.assertGreater(stats['unique_compounds'], 0)
        
        # Test data retrieval
        for compound_info in stats['top_compounds'][:3]:
            compound_name = compound_info[0]
            results = self.db.search_by_compound_name(compound_name, exact_match=True)
            self.assertGreater(len(results), 0)
            
            # Test spectrum retrieval
            spectrum_data = self.db.get_spectrum_by_id(results[0]['id'])
            self.assertIsNotNone(spectrum_data)
            self.assertIn('spectrum_data', spectrum_data)
            self.assertIsInstance(spectrum_data['spectrum_data'], np.ndarray)


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAdvancedPreprocessor,
        TestPeakMatcher,
        TestRamanSpectralDatabase,
        TestEnsembleClassifier,
        TestConfidenceScorer,
        TestSyntheticDataGenerator,
        TestDataIntegrator,
        TestValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    print("Running Tonkatsu-OS Test Suite...")
    print("=" * 50)
    
    result = run_all_tests()
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")