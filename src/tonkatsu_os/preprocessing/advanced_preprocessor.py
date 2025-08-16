"""
Advanced Raman Spectral Preprocessing Pipeline

This module provides comprehensive preprocessing capabilities for Raman spectra
including noise reduction, baseline correction, normalization, and peak detection.
"""

import numpy as np
from scipy import signal
from scipy.signal import savgol_filter, find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline for Raman spectra with multiple
    noise reduction and normalization techniques.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        
    def preprocess(self, spectrum: np.ndarray, 
                  smooth: bool = True,
                  baseline_correct: bool = True,
                  normalize: bool = True,
                  remove_spikes: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for Raman spectrum.
        
        Args:
            spectrum: Raw spectrum data
            smooth: Apply Savitzky-Golay smoothing
            baseline_correct: Apply baseline correction
            normalize: Normalize the spectrum
            remove_spikes: Remove cosmic ray spikes
            
        Returns:
            Preprocessed spectrum
        """
        processed = spectrum.copy().astype(float)
        
        # Remove cosmic ray spikes
        if remove_spikes:
            processed = self.remove_cosmic_rays(processed)
        
        # Smooth the spectrum
        if smooth:
            processed = self.smooth_spectrum(processed)
        
        # Baseline correction
        if baseline_correct:
            processed = self.baseline_correction(processed)
        
        # Normalization
        if normalize:
            processed = self.normalize_spectrum(processed)
        
        return processed
    
    def smooth_spectrum(self, spectrum: np.ndarray, 
                       window_length: int = 11, 
                       polyorder: int = 3) -> np.ndarray:
        """Apply Savitzky-Golay smoothing filter."""
        if len(spectrum) < window_length:
            window_length = len(spectrum) if len(spectrum) % 2 == 1 else len(spectrum) - 1
        
        return savgol_filter(spectrum, window_length, polyorder)
    
    def baseline_correction(self, spectrum: np.ndarray, 
                          lam: float = 1e6, 
                          p: float = 0.001,
                          niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares (ALS) baseline correction.
        
        Args:
            spectrum: Input spectrum
            lam: Smoothness parameter
            p: Asymmetry parameter
            niter: Number of iterations
            
        Returns:
            Baseline-corrected spectrum
        """
        L = len(spectrum)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        
        w = np.ones(L)
        W = diags(w, 0, shape=(L, L))
        
        for i in range(niter):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w * spectrum)
            w = p * (spectrum > z) + (1-p) * (spectrum < z)
        
        return spectrum - z
    
    def normalize_spectrum(self, spectrum: np.ndarray, 
                          method: str = 'minmax') -> np.ndarray:
        """
        Normalize spectrum using various methods.
        
        Args:
            spectrum: Input spectrum
            method: 'minmax', 'standard', 'l2', 'area'
            
        Returns:
            Normalized spectrum
        """
        if method == 'minmax':
            return (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
        elif method == 'standard':
            return (spectrum - np.mean(spectrum)) / np.std(spectrum)
        elif method == 'l2':
            return spectrum / np.linalg.norm(spectrum)
        elif method == 'area':
            return spectrum / np.trapz(spectrum)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def remove_cosmic_rays(self, spectrum: np.ndarray, 
                          threshold: float = 5.0,
                          window_size: int = 5) -> np.ndarray:
        """
        Remove cosmic ray spikes using median filtering and threshold detection.
        
        Args:
            spectrum: Input spectrum
            threshold: Threshold for spike detection (standard deviations)
            window_size: Size of median filter window
            
        Returns:
            Spectrum with cosmic rays removed
        """
        # Apply median filter
        filtered = signal.medfilt(spectrum, kernel_size=window_size)
        
        # Calculate residuals
        residuals = spectrum - filtered
        
        # Identify spikes
        std_residuals = np.std(residuals)
        spikes = np.abs(residuals) > threshold * std_residuals
        
        # Replace spikes with filtered values
        corrected = spectrum.copy()
        corrected[spikes] = filtered[spikes]
        
        return corrected
    
    def detect_peaks(self, spectrum: np.ndarray,
                    height: float = None,
                    prominence: float = 0.01,
                    distance: int = 10,
                    width: int = None) -> tuple:
        """
        Detect peaks in the spectrum using scipy.signal.find_peaks.
        
        Args:
            spectrum: Input spectrum
            height: Minimum peak height
            prominence: Minimum peak prominence
            distance: Minimum distance between peaks
            width: Peak width constraints
            
        Returns:
            Tuple of (peak_positions, peak_properties)
        """
        if height is None:
            height = np.max(spectrum) * 0.05  # 5% of max intensity
        
        peaks, properties = find_peaks(
            spectrum,
            height=height,
            prominence=prominence,
            distance=distance,
            width=width
        )
        
        # Extract peak intensities
        peak_intensities = spectrum[peaks]
        
        return peaks, peak_intensities
    
    def calculate_peak_properties(self, spectrum: np.ndarray, 
                                 peaks: np.ndarray) -> dict:
        """
        Calculate detailed properties for detected peaks.
        
        Args:
            spectrum: Input spectrum
            peaks: Peak positions
            
        Returns:
            Dictionary with peak properties
        """
        properties = {
            'positions': peaks,
            'intensities': spectrum[peaks],
            'heights': spectrum[peaks],
            'areas': [],
            'widths': [],
            'fwhm': []
        }
        
        for peak in peaks:
            # Calculate peak area (simple triangular approximation)
            left = max(0, peak - 10)
            right = min(len(spectrum), peak + 10)
            area = np.trapz(spectrum[left:right])
            properties['areas'].append(area)
            
            # Calculate FWHM
            half_max = spectrum[peak] / 2
            
            # Find left and right half-maximum points
            left_idx = peak
            while left_idx > 0 and spectrum[left_idx] > half_max:
                left_idx -= 1
            
            right_idx = peak
            while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
                right_idx += 1
            
            fwhm = right_idx - left_idx
            properties['fwhm'].append(fwhm)
            properties['widths'].append(fwhm)
        
        # Convert to numpy arrays
        for key in ['areas', 'widths', 'fwhm']:
            properties[key] = np.array(properties[key])
        
        return properties
    
    def spectral_features(self, spectrum: np.ndarray) -> dict:
        """
        Extract comprehensive spectral features for ML classification.
        
        Args:
            spectrum: Preprocessed spectrum
            
        Returns:
            Dictionary of spectral features
        """
        peaks, peak_intensities = self.detect_peaks(spectrum)
        
        features = {
            # Basic statistics
            'mean_intensity': np.mean(spectrum),
            'std_intensity': np.std(spectrum),
            'max_intensity': np.max(spectrum),
            'min_intensity': np.min(spectrum),
            'intensity_range': np.max(spectrum) - np.min(spectrum),
            
            # Peak-related features
            'num_peaks': len(peaks),
            'peak_density': len(peaks) / len(spectrum),
            'dominant_peak_position': peaks[np.argmax(peak_intensities)] if len(peaks) > 0 else 0,
            'dominant_peak_intensity': np.max(peak_intensities) if len(peaks) > 0 else 0,
            'mean_peak_intensity': np.mean(peak_intensities) if len(peaks) > 0 else 0,
            
            # Spectral moments
            'spectral_centroid': self._spectral_centroid(spectrum),
            'spectral_spread': self._spectral_spread(spectrum),
            'spectral_skewness': self._spectral_skewness(spectrum),
            'spectral_kurtosis': self._spectral_kurtosis(spectrum),
            
            # Energy distribution
            'energy_below_1000': np.sum(spectrum[:1000]) if len(spectrum) > 1000 else np.sum(spectrum),
            'energy_1000_1500': np.sum(spectrum[1000:1500]) if len(spectrum) > 1500 else 0,
            'energy_above_1500': np.sum(spectrum[1500:]) if len(spectrum) > 1500 else 0,
        }
        
        return features
    
    def _spectral_centroid(self, spectrum: np.ndarray) -> float:
        """Calculate spectral centroid (center of mass)."""
        freqs = np.arange(len(spectrum))
        return np.sum(freqs * spectrum) / np.sum(spectrum)
    
    def _spectral_spread(self, spectrum: np.ndarray) -> float:
        """Calculate spectral spread (standard deviation around centroid)."""
        freqs = np.arange(len(spectrum))
        centroid = self._spectral_centroid(spectrum)
        return np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum))
    
    def _spectral_skewness(self, spectrum: np.ndarray) -> float:
        """Calculate spectral skewness."""
        freqs = np.arange(len(spectrum))
        centroid = self._spectral_centroid(spectrum)
        spread = self._spectral_spread(spectrum)
        
        if spread == 0:
            return 0
        
        return np.sum(((freqs - centroid) ** 3) * spectrum) / (np.sum(spectrum) * spread ** 3)
    
    def _spectral_kurtosis(self, spectrum: np.ndarray) -> float:
        """Calculate spectral kurtosis."""
        freqs = np.arange(len(spectrum))
        centroid = self._spectral_centroid(spectrum)
        spread = self._spectral_spread(spectrum)
        
        if spread == 0:
            return 0
        
        return np.sum(((freqs - centroid) ** 4) * spectrum) / (np.sum(spectrum) * spread ** 4) - 3


class PeakMatcher:
    """
    Advanced peak matching system for identifying characteristic Raman shifts.
    """
    
    def __init__(self, tolerance: float = 5.0):
        """
        Initialize peak matcher.
        
        Args:
            tolerance: Tolerance for peak matching in wavenumber units
        """
        self.tolerance = tolerance
        self.reference_peaks = {}
    
    def add_reference_peaks(self, compound_name: str, peaks: list):
        """Add reference peaks for a compound."""
        self.reference_peaks[compound_name] = np.array(peaks)
    
    def match_peaks(self, observed_peaks: np.ndarray, 
                   reference_peaks: np.ndarray) -> dict:
        """
        Match observed peaks with reference peaks.
        
        Args:
            observed_peaks: Observed peak positions
            reference_peaks: Reference peak positions
            
        Returns:
            Dictionary with matching results
        """
        matches = []
        unmatched_observed = list(observed_peaks)
        unmatched_reference = list(reference_peaks)
        
        for obs_peak in observed_peaks:
            for ref_peak in reference_peaks:
                if abs(obs_peak - ref_peak) <= self.tolerance:
                    matches.append({
                        'observed': obs_peak,
                        'reference': ref_peak,
                        'difference': abs(obs_peak - ref_peak)
                    })
                    if obs_peak in unmatched_observed:
                        unmatched_observed.remove(obs_peak)
                    if ref_peak in unmatched_reference:
                        unmatched_reference.remove(ref_peak)
                    break
        
        match_score = len(matches) / max(len(observed_peaks), len(reference_peaks))
        
        return {
            'matches': matches,
            'match_score': match_score,
            'unmatched_observed': unmatched_observed,
            'unmatched_reference': unmatched_reference,
            'num_matches': len(matches)
        }
    
    def identify_compound(self, observed_peaks: np.ndarray) -> list:
        """
        Identify the most likely compounds based on peak matching.
        
        Args:
            observed_peaks: Observed peak positions
            
        Returns:
            List of candidate compounds with match scores
        """
        candidates = []
        
        for compound, ref_peaks in self.reference_peaks.items():
            match_result = self.match_peaks(observed_peaks, ref_peaks)
            
            candidates.append({
                'compound': compound,
                'match_score': match_result['match_score'],
                'num_matches': match_result['num_matches'],
                'matches': match_result['matches']
            })
        
        # Sort by match score
        candidates.sort(key=lambda x: x['match_score'], reverse=True)
        
        return candidates