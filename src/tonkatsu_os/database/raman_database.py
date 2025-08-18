"""
Raman Spectra Database Management System

This module provides a comprehensive database system for storing, indexing, and searching
Raman spectra with vectorized representations for efficient similarity matching.
"""

import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RamanSpectralDatabase:
    """
    A comprehensive database system for Raman spectra with vectorized storage
    and efficient similarity search capabilities.
    """

    def __init__(self, db_path: str = "raman_spectra.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self.scaler = StandardScaler()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the SQLite database with required tables."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        # Create main spectra table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spectra (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                compound_name TEXT NOT NULL,
                chemical_formula TEXT,
                cas_number TEXT,
                measurement_conditions TEXT,
                laser_wavelength REAL,
                integration_time REAL,
                acquisition_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                spectrum_data BLOB NOT NULL,
                preprocessed_spectrum BLOB,
                peak_positions BLOB,
                peak_intensities BLOB,
                spectral_fingerprint BLOB,
                metadata TEXT
            )
        """
        )

        # Create index for faster compound name searches
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_compound_name 
            ON spectra(compound_name)
        """
        )

        # Create table for spectral features (vectorized representations)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spectral_features (
                spectrum_id INTEGER PRIMARY KEY,
                feature_vector BLOB NOT NULL,
                pca_components BLOB,
                dominant_peaks BLOB,
                spectral_hash TEXT,
                FOREIGN KEY (spectrum_id) REFERENCES spectra(id)
            )
        """
        )

        # Create table for classification results
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                spectrum_id INTEGER,
                predicted_compound TEXT,
                confidence_score REAL,
                model_used TEXT,
                classification_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (spectrum_id) REFERENCES spectra(id)
            )
        """
        )

        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def add_spectrum(
        self,
        compound_name: str,
        spectrum_data: np.ndarray,
        chemical_formula: str = None,
        cas_number: str = None,
        measurement_conditions: str = None,
        laser_wavelength: float = 473.0,
        integration_time: float = 200.0,
        metadata: dict = None,
    ) -> int:
        """
        Add a new spectrum to the database with automatic feature extraction.

        Args:
            compound_name: Name of the compound
            spectrum_data: Raw spectrum data as numpy array
            chemical_formula: Chemical formula (optional)
            cas_number: CAS registry number (optional)
            measurement_conditions: Description of measurement conditions
            laser_wavelength: Laser wavelength in nm
            integration_time: Integration time in ms
            metadata: Additional metadata as dictionary

        Returns:
            int: Database ID of the added spectrum
        """
        from tonkatsu_os.preprocessing import AdvancedPreprocessor

        # Preprocess the spectrum
        preprocessor = AdvancedPreprocessor()
        processed_spectrum = preprocessor.preprocess(spectrum_data)
        peaks, peak_intensities = preprocessor.detect_peaks(processed_spectrum)

        # Generate spectral fingerprint (key features)
        fingerprint = self._generate_fingerprint(processed_spectrum, peaks, peak_intensities)

        # Insert into main table
        cursor = self.conn.execute(
            """
            INSERT INTO spectra (
                compound_name, chemical_formula, cas_number, 
                measurement_conditions, laser_wavelength, integration_time,
                spectrum_data, preprocessed_spectrum, peak_positions, 
                peak_intensities, spectral_fingerprint, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                compound_name,
                chemical_formula,
                cas_number,
                measurement_conditions,
                laser_wavelength,
                integration_time,
                pickle.dumps(spectrum_data),
                pickle.dumps(processed_spectrum),
                pickle.dumps(peaks),
                pickle.dumps(peak_intensities),
                pickle.dumps(fingerprint),
                str(metadata) if metadata else None,
            ),
        )

        spectrum_id = cursor.lastrowid

        # Generate and store feature vector
        feature_vector = self._extract_features(processed_spectrum, peaks, peak_intensities)
        spectral_hash = self._compute_spectral_hash(processed_spectrum)

        self.conn.execute(
            """
            INSERT INTO spectral_features (
                spectrum_id, feature_vector, dominant_peaks, spectral_hash
            ) VALUES (?, ?, ?, ?)
        """,
            (
                spectrum_id,
                pickle.dumps(feature_vector),
                pickle.dumps(peaks[:10]),
                spectral_hash,  # Store top 10 peaks
            ),
        )

        self.conn.commit()
        logger.info(f"Added spectrum for {compound_name} with ID {spectrum_id}")
        return spectrum_id

    def search_similar_spectra(
        self, query_spectrum: np.ndarray, top_k: int = 10, similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Find similar spectra using cosine similarity on feature vectors.

        Args:
            query_spectrum: Query spectrum as numpy array
            top_k: Number of top similar spectra to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of dictionaries with spectrum info and similarity scores
        """
        from tonkatsu_os.preprocessing import AdvancedPreprocessor

        # Preprocess query spectrum
        preprocessor = AdvancedPreprocessor()
        processed_query = preprocessor.preprocess(query_spectrum)
        peaks, peak_intensities = preprocessor.detect_peaks(processed_query)
        query_features = self._extract_features(processed_query, peaks, peak_intensities)

        # Get all feature vectors from database
        cursor = self.conn.execute(
            """
            SELECT sf.spectrum_id, sf.feature_vector, s.compound_name, 
                   s.chemical_formula, s.cas_number
            FROM spectral_features sf
            JOIN spectra s ON sf.spectrum_id = s.id
        """
        )

        results = []
        for row in cursor.fetchall():
            spectrum_id, feature_blob, compound_name, formula, cas_number = row
            db_features = pickle.loads(feature_blob)

            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_features.reshape(1, -1), db_features.reshape(1, -1)
            )[0, 0]

            if similarity >= similarity_threshold:
                results.append(
                    {
                        "spectrum_id": spectrum_id,
                        "compound_name": compound_name,
                        "chemical_formula": formula,
                        "cas_number": cas_number,
                        "similarity_score": similarity,
                    }
                )

        # Sort by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def get_spectrum_by_id(self, spectrum_id: int) -> Optional[Dict]:
        """Retrieve a spectrum by its database ID."""
        cursor = self.conn.execute(
            """
            SELECT * FROM spectra WHERE id = ?
        """,
            (spectrum_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Convert to dictionary
        columns = [desc[0] for desc in cursor.description]
        spectrum_dict = dict(zip(columns, row))

        # Deserialize binary data
        spectrum_dict["spectrum_data"] = pickle.loads(spectrum_dict["spectrum_data"])
        if spectrum_dict["preprocessed_spectrum"]:
            spectrum_dict["preprocessed_spectrum"] = pickle.loads(
                spectrum_dict["preprocessed_spectrum"]
            )
        if spectrum_dict["peak_positions"]:
            spectrum_dict["peak_positions"] = pickle.loads(spectrum_dict["peak_positions"])
        if spectrum_dict["peak_intensities"]:
            spectrum_dict["peak_intensities"] = pickle.loads(spectrum_dict["peak_intensities"])
        if spectrum_dict["spectral_fingerprint"]:
            spectrum_dict["spectral_fingerprint"] = pickle.loads(
                spectrum_dict["spectral_fingerprint"]
            )

        return spectrum_dict

    def search_by_compound_name(self, compound_name: str, exact_match: bool = False) -> List[Dict]:
        """Search spectra by compound name."""
        if exact_match:
            query = (
                "SELECT id, compound_name, chemical_formula FROM spectra WHERE compound_name = ?"
            )
            params = (compound_name,)
        else:
            query = (
                "SELECT id, compound_name, chemical_formula FROM spectra WHERE compound_name LIKE ?"
            )
            params = (f"%{compound_name}%",)

        cursor = self.conn.execute(query, params)
        return [
            {"id": row[0], "compound_name": row[1], "chemical_formula": row[2]}
            for row in cursor.fetchall()
        ]

    def get_database_stats(self) -> Dict:
        """Get statistics about the database."""
        stats = {}

        # Total number of spectra
        cursor = self.conn.execute("SELECT COUNT(*) FROM spectra")
        stats["total_spectra"] = cursor.fetchone()[0]

        # Unique compounds
        cursor = self.conn.execute("SELECT COUNT(DISTINCT compound_name) FROM spectra")
        stats["unique_compounds"] = cursor.fetchone()[0]

        # Most common compounds
        cursor = self.conn.execute(
            """
            SELECT compound_name, COUNT(*) as count 
            FROM spectra 
            GROUP BY compound_name 
            ORDER BY count DESC 
            LIMIT 10
        """
        )
        stats["top_compounds"] = cursor.fetchall()

        return stats

    def _extract_features(
        self, spectrum: np.ndarray, peaks: np.ndarray, peak_intensities: np.ndarray
    ) -> np.ndarray:
        """Extract feature vector from preprocessed spectrum."""
        features = []

        # Statistical features
        features.extend(
            [
                np.mean(spectrum),
                np.std(spectrum),
                np.max(spectrum),
                np.min(spectrum),
                len(peaks),  # Number of peaks
                np.mean(peak_intensities) if len(peak_intensities) > 0 else 0,
                np.std(peak_intensities) if len(peak_intensities) > 0 else 0,
            ]
        )

        # Spectral moments
        x = np.arange(len(spectrum))
        features.extend(
            [
                np.sum(x * spectrum) / np.sum(spectrum),  # Centroid
                np.sqrt(np.sum(((x - features[-1]) ** 2) * spectrum) / np.sum(spectrum)),  # Spread
            ]
        )

        # Binned intensities (reduce dimensionality)
        n_bins = 50
        binned = np.histogram(spectrum, bins=n_bins)[0]
        features.extend(binned.tolist())

        # Top peak positions (normalized)
        top_peaks = (
            peaks[:10] if len(peaks) >= 10 else np.pad(peaks, (0, 10 - len(peaks)), "constant")
        )
        features.extend((top_peaks / len(spectrum)).tolist())

        return np.array(features)

    def _generate_fingerprint(
        self, spectrum: np.ndarray, peaks: np.ndarray, peak_intensities: np.ndarray
    ) -> Dict:
        """Generate a compact spectral fingerprint for quick matching."""
        return {
            "dominant_peak": int(peaks[0]) if len(peaks) > 0 else None,
            "peak_count": len(peaks),
            "intensity_range": float(np.max(spectrum) - np.min(spectrum)),
            "spectral_centroid": float(
                np.sum(np.arange(len(spectrum)) * spectrum) / np.sum(spectrum)
            ),
            "top_5_peaks": peaks[:5].tolist() if len(peaks) > 0 else [],
        }

    def _compute_spectral_hash(self, spectrum: np.ndarray) -> str:
        """Compute a hash for quick duplicate detection."""
        import hashlib

        # Use a subset of the spectrum for hashing
        subset = spectrum[::10]  # Every 10th point
        return hashlib.md5(subset.tobytes()).hexdigest()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
