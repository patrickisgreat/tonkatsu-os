"""
Data Loader for Public Raman Spectral Databases

This module provides functionality to download, parse, and integrate
public Raman spectral databases like RRUFF and NIST into the local database.
"""

import requests
import numpy as np
import pandas as pd
from pathlib import Path
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from tonkatsu_os.database import RamanSpectralDatabase
from tonkatsu_os.preprocessing import AdvancedPreprocessor

logger = logging.getLogger(__name__)


class RRUFFDataLoader:
    """
    Loader for RRUFF (Raman, X-ray, Infrared, and Chemistry) database.
    """
    
    def __init__(self, data_dir: str = "rruff_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.base_url = "https://rruff.info"
        
    def download_rruff_list(self) -> pd.DataFrame:
        """Download the complete RRUFF mineral list."""
        try:
            # Download the mineral list
            list_url = f"{self.base_url}/zipped_data_files/rruff.txt"
            response = requests.get(list_url, timeout=30)
            response.raise_for_status()
            
            # Save the file
            list_file = self.data_dir / "rruff_list.txt"
            with open(list_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Parse into DataFrame
            lines = response.text.strip().split('\n')
            data = []
            
            for line in lines[1:]:  # Skip header
                fields = line.split('\t')
                if len(fields) >= 4:
                    data.append({
                        'rruff_id': fields[0],
                        'mineral_name': fields[1],
                        'chemistry': fields[2] if len(fields) > 2 else '',
                        'locality': fields[3] if len(fields) > 3 else ''
                    })
            
            df = pd.DataFrame(data)
            logger.info(f"Downloaded {len(df)} RRUFF entries")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading RRUFF list: {e}")
            return pd.DataFrame()
    
    def download_spectrum(self, rruff_id: str, spectrum_type: str = "oriented") -> Optional[np.ndarray]:
        """
        Download a specific Raman spectrum from RRUFF.
        
        Args:
            rruff_id: RRUFF database ID
            spectrum_type: Type of spectrum ('oriented', 'unoriented', 'laser')
            
        Returns:
            Spectrum data as numpy array or None if failed
        """
        try:
            spectrum_url = f"{self.base_url}/tmp_rruff/{rruff_id}_{spectrum_type}__rruff.txt"
            response = requests.get(spectrum_url, timeout=15)
            
            if response.status_code == 200:
                # Parse the spectrum data
                lines = response.text.strip().split('\n')
                
                # Find the data section (after header)
                data_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('##'):  # JCAMP-DX format
                        continue
                    if ',' in line or '\t' in line:
                        data_start = i
                        break
                
                # Extract wavelength and intensity data
                wavelengths = []
                intensities = []
                
                for line in lines[data_start:]:
                    if line.strip() and not line.startswith('#'):
                        parts = re.split(r'[,\t\s]+', line.strip())
                        if len(parts) >= 2:
                            try:
                                wavelength = float(parts[0])
                                intensity = float(parts[1])
                                wavelengths.append(wavelength)
                                intensities.append(intensity)
                            except ValueError:
                                continue
                
                if len(intensities) > 0:
                    return np.array(intensities)
                
        except Exception as e:
            logger.warning(f"Failed to download spectrum {rruff_id}: {e}")
        
        return None
    
    def batch_download_spectra(self, mineral_list: pd.DataFrame, 
                              max_spectra: int = 100) -> List[Dict]:
        """
        Download multiple spectra in batch.
        
        Args:
            mineral_list: DataFrame with RRUFF entries
            max_spectra: Maximum number of spectra to download
            
        Returns:
            List of successfully downloaded spectra
        """
        downloaded_spectra = []
        
        for i, row in mineral_list.head(max_spectra).iterrows():
            rruff_id = row['rruff_id']
            mineral_name = row['mineral_name']
            chemistry = row['chemistry']
            
            logger.info(f"Downloading spectrum {i+1}/{max_spectra}: {mineral_name}")
            
            spectrum = self.download_spectrum(rruff_id)
            
            if spectrum is not None:
                downloaded_spectra.append({
                    'rruff_id': rruff_id,
                    'compound_name': mineral_name,
                    'chemical_formula': chemistry,
                    'spectrum_data': spectrum,
                    'source': 'RRUFF',
                    'measurement_conditions': 'RRUFF database standard conditions'
                })
                
                logger.info(f"Successfully downloaded {mineral_name}")
            else:
                logger.warning(f"Failed to download {mineral_name}")
        
        logger.info(f"Downloaded {len(downloaded_spectra)} spectra successfully")
        return downloaded_spectra


class NISTDataLoader:
    """
    Loader for NIST Raman spectral data.
    Note: NIST data may require different access methods.
    """
    
    def __init__(self, data_dir: str = "nist_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_local_nist_data(self, file_path: str) -> List[Dict]:
        """
        Load NIST data from local files.
        
        Args:
            file_path: Path to NIST data file
            
        Returns:
            List of spectral data dictionaries
        """
        spectra = []
        
        try:
            # This is a placeholder - actual implementation depends on NIST file format
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                
                for _, row in df.iterrows():
                    if 'compound_name' in row and 'spectrum' in row:
                        spectra.append({
                            'compound_name': row['compound_name'],
                            'chemical_formula': row.get('formula', ''),
                            'spectrum_data': np.array(eval(row['spectrum'])),
                            'source': 'NIST',
                            'cas_number': row.get('cas_number', ''),
                            'measurement_conditions': row.get('conditions', 'NIST standard')
                        })
            
        except Exception as e:
            logger.error(f"Error loading NIST data: {e}")
        
        return spectra


class SyntheticDataGenerator:
    """
    Generate synthetic Raman spectra for testing and training.
    """
    
    def __init__(self):
        self.common_compounds = {
            'water': {'peaks': [3200, 3400], 'formula': 'H2O'},
            'ethanol': {'peaks': [880, 1050, 1450, 2900, 3000], 'formula': 'C2H5OH'},
            'benzene': {'peaks': [992, 1178, 1595, 3047], 'formula': 'C6H6'},
            'acetone': {'peaks': [790, 1430, 1700, 2900], 'formula': 'C3H6O'},
            'glucose': {'peaks': [1126, 1365, 1462, 2900], 'formula': 'C6H12O6'}
        }
    
    def generate_synthetic_spectrum(self, compound_name: str, 
                                  length: int = 2048,
                                  noise_level: float = 0.1) -> np.ndarray:
        """
        Generate a synthetic Raman spectrum for a compound.
        
        Args:
            compound_name: Name of the compound
            length: Length of the spectrum
            noise_level: Amount of noise to add
            
        Returns:
            Synthetic spectrum
        """
        spectrum = np.zeros(length)
        
        if compound_name in self.common_compounds:
            peaks = self.common_compounds[compound_name]['peaks']
            
            for peak_pos in peaks:
                if peak_pos < length:
                    # Add Gaussian peak
                    peak_width = np.random.uniform(10, 30)
                    peak_intensity = np.random.uniform(0.5, 1.0)
                    
                    x = np.arange(length)
                    peak = peak_intensity * np.exp(-((x - peak_pos) ** 2) / (2 * peak_width ** 2))
                    spectrum += peak
        
        # Add noise
        noise = np.random.normal(0, noise_level, length)
        spectrum += noise
        
        # Add baseline
        baseline = np.random.uniform(0.1, 0.3)
        spectrum += baseline
        
        return np.maximum(spectrum, 0)  # Ensure non-negative
    
    def generate_training_dataset(self, n_samples_per_compound: int = 10) -> List[Dict]:
        """
        Generate a synthetic training dataset.
        
        Args:
            n_samples_per_compound: Number of samples per compound
            
        Returns:
            List of synthetic spectra
        """
        dataset = []
        
        for compound_name, info in self.common_compounds.items():
            for i in range(n_samples_per_compound):
                spectrum = self.generate_synthetic_spectrum(compound_name)
                
                dataset.append({
                    'compound_name': compound_name,
                    'chemical_formula': info['formula'],
                    'spectrum_data': spectrum,
                    'source': 'synthetic',
                    'measurement_conditions': f'synthetic_sample_{i+1}'
                })
        
        logger.info(f"Generated {len(dataset)} synthetic spectra")
        return dataset


class DataIntegrator:
    """
    Integrate data from multiple sources into the main database.
    """
    
    def __init__(self, database: RamanSpectralDatabase):
        self.database = database
        self.preprocessor = AdvancedPreprocessor()
    
    def integrate_spectra(self, spectra_list: List[Dict]) -> Dict:
        """
        Integrate a list of spectra into the database.
        
        Args:
            spectra_list: List of spectrum dictionaries
            
        Returns:
            Integration results
        """
        success_count = 0
        error_count = 0
        errors = []
        
        for spectrum_dict in spectra_list:
            try:
                # Validate required fields
                required_fields = ['compound_name', 'spectrum_data']
                if not all(field in spectrum_dict for field in required_fields):
                    raise ValueError(f"Missing required fields: {required_fields}")
                
                # Add to database
                spectrum_id = self.database.add_spectrum(
                    compound_name=spectrum_dict['compound_name'],
                    spectrum_data=spectrum_dict['spectrum_data'],
                    chemical_formula=spectrum_dict.get('chemical_formula', ''),
                    cas_number=spectrum_dict.get('cas_number', ''),
                    measurement_conditions=spectrum_dict.get('measurement_conditions', ''),
                    laser_wavelength=spectrum_dict.get('laser_wavelength', 473.0),
                    integration_time=spectrum_dict.get('integration_time', 200.0),
                    metadata={
                        'source': spectrum_dict.get('source', 'unknown'),
                        'original_id': spectrum_dict.get('rruff_id', '')
                    }
                )
                
                success_count += 1
                logger.info(f"Added spectrum ID {spectrum_id} for {spectrum_dict['compound_name']}")
                
            except Exception as e:
                error_count += 1
                error_msg = f"Error adding {spectrum_dict.get('compound_name', 'unknown')}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        results = {
            'total_processed': len(spectra_list),
            'successful_integrations': success_count,
            'errors': error_count,
            'error_messages': errors
        }
        
        logger.info(f"Integration complete: {success_count} successful, {error_count} errors")
        return results
    
    def download_and_integrate_rruff(self, max_spectra: int = 50) -> Dict:
        """
        Download RRUFF data and integrate into database.
        
        Args:
            max_spectra: Maximum number of spectra to download
            
        Returns:
            Integration results
        """
        logger.info("Starting RRUFF data download and integration...")
        
        rruff_loader = RRUFFDataLoader()
        
        # Download mineral list
        mineral_list = rruff_loader.download_rruff_list()
        
        if mineral_list.empty:
            return {'error': 'Failed to download RRUFF mineral list'}
        
        # Download spectra
        spectra = rruff_loader.batch_download_spectra(mineral_list, max_spectra)
        
        # Integrate into database
        return self.integrate_spectra(spectra)
    
    def generate_and_integrate_synthetic(self, n_samples_per_compound: int = 10) -> Dict:
        """
        Generate synthetic data and integrate into database.
        
        Args:
            n_samples_per_compound: Number of samples per compound
            
        Returns:
            Integration results
        """
        logger.info("Generating and integrating synthetic data...")
        
        generator = SyntheticDataGenerator()
        synthetic_data = generator.generate_training_dataset(n_samples_per_compound)
        
        return self.integrate_spectra(synthetic_data)