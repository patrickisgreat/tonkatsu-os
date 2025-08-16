"""
Spectrum Import Module with Multiple Format Support

This module provides comprehensive spectrum import capabilities with support
for various file formats, batch processing, and interactive validation.
"""

import pandas as pd
import numpy as np
import streamlit as st
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from tonkatsu_os.database import RamanSpectralDatabase
from tonkatsu_os.preprocessing import AdvancedPreprocessor

logger = logging.getLogger(__name__)


class SpectrumImporter:
    """
    Comprehensive spectrum import system supporting multiple file formats
    and validation workflows.
    """
    
    def __init__(self, database: RamanSpectralDatabase):
        self.database = database
        self.preprocessor = AdvancedPreprocessor()
        self.supported_formats = [
            '.csv', '.txt', '.json', '.xlsx', '.tsv'
        ]
    
    def create_import_interface(self):
        """Create Streamlit interface for spectrum import."""
        st.header("📥 Spectrum Import Center")
        
        # Import method selection
        import_method = st.radio(
            "Choose import method:",
            ["📁 File Upload", "📋 Paste Data", "🔗 URL Import", "📊 Batch Import"]
        )
        
        if import_method == "📁 File Upload":
            self._file_upload_interface()
        elif import_method == "📋 Paste Data":
            self._paste_data_interface()
        elif import_method == "🔗 URL Import":
            self._url_import_interface()
        elif import_method == "📊 Batch Import":
            self._batch_import_interface()
    
    def _file_upload_interface(self):
        """File upload interface with drag and drop."""
        st.subheader("📁 Upload Spectrum Files")
        
        # File uploader with multiple format support
        uploaded_files = st.file_uploader(
            "Drag and drop files here or click to browse",
            type=['csv', 'txt', 'json', 'xlsx', 'tsv'],
            accept_multiple_files=True,
            help="Supported formats: CSV, TXT, JSON, XLSX, TSV"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")
            
            # Process each file
            processed_spectra = []
            for uploaded_file in uploaded_files:
                with st.expander(f"📄 {uploaded_file.name}"):
                    try:
                        spectrum_data = self._parse_uploaded_file(uploaded_file)
                        if spectrum_data:
                            processed_spectra.append(spectrum_data)
                            self._preview_spectrum(spectrum_data, uploaded_file.name)
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            # Batch import confirmation
            if processed_spectra:
                self._import_confirmation_interface(processed_spectra)
    
    def _paste_data_interface(self):
        """Interface for pasting spectrum data directly."""
        st.subheader("📋 Paste Spectrum Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Data input area
            data_input = st.text_area(
                "Paste your spectrum data here:",
                height=200,
                help="Supported formats: CSV, tab-separated, space-separated"
            )
            
            # Format specification
            data_format = st.selectbox(
                "Data format:",
                ["Auto-detect", "CSV (comma-separated)", "Tab-separated", "Space-separated", "JSON"]
            )
        
        with col2:
            # Metadata input
            st.subheader("Spectrum Information")
            compound_name = st.text_input("Compound Name*", key="paste_compound")
            chemical_formula = st.text_input("Chemical Formula", key="paste_formula")
            cas_number = st.text_input("CAS Number", key="paste_cas")
            source = st.text_input("Data Source", key="paste_source")
            
            # Processing options
            st.subheader("Processing Options")
            auto_preprocess = st.checkbox("Auto-preprocess spectrum", value=True)
            validate_data = st.checkbox("Validate data quality", value=True)
        
        if st.button("🔍 Parse and Preview", type="primary") and data_input and compound_name:
            try:
                # Parse the input data
                spectrum_data = self._parse_text_data(data_input, data_format)
                
                if spectrum_data is not None:
                    # Create spectrum info
                    spectrum_info = {
                        'compound_name': compound_name,
                        'chemical_formula': chemical_formula,
                        'cas_number': cas_number,
                        'source': source or 'manual_input',
                        'spectrum_data': spectrum_data,
                        'auto_preprocess': auto_preprocess,
                        'validate_data': validate_data
                    }
                    
                    # Preview and validate
                    self._preview_spectrum(spectrum_info, "Pasted Data")
                    
                    # Store in session state for import
                    if 'import_queue' not in st.session_state:
                        st.session_state.import_queue = []
                    st.session_state.import_queue.append(spectrum_info)
                    
                    if st.button("✅ Add to Import Queue"):
                        st.success("Spectrum added to import queue!")
                
            except Exception as e:
                st.error(f"❌ Error parsing data: {str(e)}")
    
    def _url_import_interface(self):
        """Interface for importing spectra from URLs."""
        st.subheader("🔗 Import from URL")
        
        url = st.text_input("Enter URL to spectrum data:")
        
        if url and st.button("📥 Import from URL"):
            try:
                import requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Try to parse the content
                content_type = response.headers.get('content-type', '').lower()
                
                if 'json' in content_type:
                    data = response.json()
                    spectrum_data = self._parse_json_spectrum(data)
                else:
                    # Assume text-based format
                    spectrum_data = self._parse_text_data(response.text, "Auto-detect")
                
                if spectrum_data is not None:
                    spectrum_info = {
                        'compound_name': f"URL_import_{url.split('/')[-1]}",
                        'source': url,
                        'spectrum_data': spectrum_data
                    }
                    
                    self._preview_spectrum(spectrum_info, "URL Import")
                    
                    if st.button("✅ Import Spectrum"):
                        self._import_single_spectrum(spectrum_info)
                
            except Exception as e:
                st.error(f"❌ Error importing from URL: {str(e)}")
    
    def _batch_import_interface(self):
        """Interface for batch importing multiple spectra."""
        st.subheader("📊 Batch Import")
        
        # ZIP file upload for batch processing
        zip_file = st.file_uploader(
            "Upload ZIP file containing multiple spectra:",
            type=['zip'],
            help="Upload a ZIP file containing CSV, TXT, or JSON spectrum files"
        )
        
        if zip_file:
            try:
                import zipfile
                
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    spectrum_files = [f for f in file_list if any(f.endswith(ext) for ext in self.supported_formats)]
                    
                    st.info(f"📁 Found {len(spectrum_files)} spectrum files in ZIP")
                    
                    if st.button("🚀 Process Batch Import"):
                        progress_bar = st.progress(0)
                        processed_count = 0
                        errors = []
                        
                        for i, filename in enumerate(spectrum_files):
                            try:
                                with zip_ref.open(filename) as file:
                                    content = file.read()
                                    
                                    # Create mock uploaded file object
                                    mock_file = io.BytesIO(content)
                                    mock_file.name = filename
                                    
                                    spectrum_data = self._parse_uploaded_file(mock_file)
                                    if spectrum_data:
                                        # Auto-generate compound name from filename
                                        compound_name = Path(filename).stem
                                        spectrum_data['compound_name'] = compound_name
                                        spectrum_data['source'] = f'batch_import_{zip_file.name}'
                                        
                                        # Import directly
                                        self._import_single_spectrum(spectrum_data)
                                        processed_count += 1
                                
                                progress_bar.progress((i + 1) / len(spectrum_files))
                                
                            except Exception as e:
                                errors.append(f"{filename}: {str(e)}")
                        
                        # Results summary
                        st.success(f"✅ Successfully imported {processed_count} spectra!")
                        
                        if errors:
                            st.warning(f"⚠️ {len(errors)} files had errors:")
                            for error in errors:
                                st.text(f"• {error}")
            
            except Exception as e:
                st.error(f"❌ Error processing ZIP file: {str(e)}")
    
    def _parse_uploaded_file(self, uploaded_file) -> Optional[Dict]:
        """Parse uploaded file and extract spectrum data."""
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        try:
            if file_extension == '.csv':
                return self._parse_csv_file(uploaded_file)
            elif file_extension in ['.txt', '.tsv']:
                return self._parse_text_file(uploaded_file)
            elif file_extension == '.json':
                return self._parse_json_file(uploaded_file)
            elif file_extension == '.xlsx':
                return self._parse_excel_file(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
        
        except Exception as e:
            st.error(f"Error parsing file: {str(e)}")
            return None
    
    def _parse_csv_file(self, file) -> Dict:
        """Parse CSV file containing spectrum data."""
        df = pd.read_csv(file)
        
        # Try different CSV formats
        if 'intensity' in df.columns.str.lower():
            # Single column intensity data
            spectrum_data = df[df.columns[df.columns.str.lower().str.contains('intensity')][0]].values
        elif len(df.columns) == 1:
            # Single column, assume it's intensity
            spectrum_data = df.iloc[:, 0].values
        elif 'wavenumber' in df.columns.str.lower() and 'intensity' in df.columns.str.lower():
            # Two-column format with wavenumber and intensity
            intensity_col = df.columns[df.columns.str.lower().str.contains('intensity')][0]
            spectrum_data = df[intensity_col].values
        else:
            # Assume numerical columns are spectrum data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                spectrum_data = df[numeric_cols].values.flatten()
            else:
                raise ValueError("No numerical data found in CSV")
        
        return {
            'spectrum_data': spectrum_data,
            'compound_name': Path(file.name).stem,
            'source': 'csv_upload',
            'original_filename': file.name
        }
    
    def _parse_text_file(self, file) -> Dict:
        """Parse text file (tab or space separated)."""
        content = file.read().decode('utf-8')
        return self._parse_text_data(content, "Auto-detect")
    
    def _parse_text_data(self, content: str, format_type: str) -> np.ndarray:
        """Parse text data in various formats."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Remove comment lines
        data_lines = [line for line in lines if not line.startswith('#') and not line.startswith('//')]
        
        spectrum_data = []
        
        for line in data_lines:
            # Determine separator
            if format_type == "CSV (comma-separated)" or (format_type == "Auto-detect" and ',' in line):
                values = line.split(',')
            elif format_type == "Tab-separated" or (format_type == "Auto-detect" and '\t' in line):
                values = line.split('\t')
            else:
                # Space-separated
                values = line.split()
            
            # Extract numerical values
            for value in values:
                try:
                    spectrum_data.append(float(value.strip()))
                except ValueError:
                    continue
        
        if not spectrum_data:
            raise ValueError("No numerical data found")
        
        return np.array(spectrum_data)
    
    def _parse_json_file(self, file) -> Dict:
        """Parse JSON file containing spectrum data."""
        data = json.load(file)
        return self._parse_json_spectrum(data)
    
    def _parse_json_spectrum(self, data: Dict) -> Dict:
        """Parse spectrum data from JSON structure."""
        # Try different JSON structures
        spectrum_data = None
        metadata = {}
        
        if 'spectrum' in data:
            spectrum_data = np.array(data['spectrum'])
        elif 'intensity' in data:
            spectrum_data = np.array(data['intensity'])
        elif 'data' in data:
            spectrum_data = np.array(data['data'])
        elif isinstance(data, list):
            spectrum_data = np.array(data)
        else:
            # Look for any array-like data
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 100:  # Assume spectrum has >100 points
                    spectrum_data = np.array(value)
                    break
        
        if spectrum_data is None:
            raise ValueError("No spectrum data found in JSON")
        
        # Extract metadata
        metadata_keys = ['compound_name', 'chemical_formula', 'cas_number', 'source', 'conditions']
        for key in metadata_keys:
            if key in data:
                metadata[key] = data[key]
        
        return {
            'spectrum_data': spectrum_data,
            **metadata
        }
    
    def _parse_excel_file(self, file) -> Dict:
        """Parse Excel file containing spectrum data."""
        df = pd.read_excel(file)
        
        # Similar logic to CSV parsing
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            spectrum_data = df[numeric_cols[0]].values  # Use first numeric column
        else:
            raise ValueError("No numerical data found in Excel file")
        
        return {
            'spectrum_data': spectrum_data,
            'compound_name': Path(file.name).stem,
            'source': 'excel_upload',
            'original_filename': file.name
        }
    
    def _preview_spectrum(self, spectrum_info: Dict, title: str):
        """Preview spectrum data with validation."""
        st.subheader(f"👀 Preview: {title}")
        
        spectrum_data = spectrum_info['spectrum_data']
        
        # Basic validation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Data Points", len(spectrum_data))
        with col2:
            st.metric("Min Intensity", f"{np.min(spectrum_data):.2f}")
        with col3:
            st.metric("Max Intensity", f"{np.max(spectrum_data):.2f}")
        
        # Quality assessment
        quality_issues = []
        
        if len(spectrum_data) < 100:
            quality_issues.append("⚠️ Very short spectrum (< 100 points)")
        
        if np.any(np.isnan(spectrum_data)):
            quality_issues.append("⚠️ Contains NaN values")
            spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
        
        if np.any(np.isinf(spectrum_data)):
            quality_issues.append("⚠️ Contains infinite values")
        
        if np.std(spectrum_data) < 0.01:
            quality_issues.append("⚠️ Very low signal variation")
        
        # Display quality assessment
        if quality_issues:
            st.warning("Data Quality Issues:")
            for issue in quality_issues:
                st.text(issue)
        else:
            st.success("✅ Data quality looks good!")
        
        # Plot preview
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(spectrum_data)
        ax.set_title(f"Spectrum Preview: {title}")
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Preprocessing preview
        if spectrum_info.get('auto_preprocess', False):
            try:
                processed = self.preprocessor.preprocess(spectrum_data)
                
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(spectrum_data)
                ax1.set_title("Original Spectrum")
                ax1.set_xlabel("Data Point")
                ax1.set_ylabel("Intensity")
                
                ax2.plot(processed)
                ax2.set_title("Preprocessed Spectrum")
                ax2.set_xlabel("Data Point")
                ax2.set_ylabel("Normalized Intensity")
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Update spectrum data with processed version
                spectrum_info['spectrum_data'] = processed
                spectrum_info['is_preprocessed'] = True
                
            except Exception as e:
                st.warning(f"Preprocessing failed: {str(e)}")
    
    def _import_confirmation_interface(self, spectra_list: List[Dict]):
        """Interface for confirming batch imports."""
        st.subheader("✅ Import Confirmation")
        
        st.info(f"Ready to import {len(spectra_list)} spectra")
        
        # Display summary table
        summary_data = []
        for i, spectrum in enumerate(spectra_list):
            summary_data.append({
                'Index': i + 1,
                'Compound': spectrum.get('compound_name', 'Unknown'),
                'Data Points': len(spectrum['spectrum_data']),
                'Source': spectrum.get('source', 'Unknown')
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary)
        
        # Import options
        col1, col2 = st.columns(2)
        
        with col1:
            overwrite_existing = st.checkbox("Overwrite existing entries", value=False)
            validate_before_import = st.checkbox("Validate before import", value=True)
        
        with col2:
            auto_preprocess = st.checkbox("Auto-preprocess all spectra", value=True)
            generate_features = st.checkbox("Generate ML features", value=True)
        
        # Import button
        if st.button("🚀 Import All Spectra", type="primary"):
            self._execute_batch_import(
                spectra_list, 
                overwrite_existing, 
                validate_before_import, 
                auto_preprocess, 
                generate_features
            )
    
    def _import_single_spectrum(self, spectrum_info: Dict):
        """Import a single spectrum into the database."""
        try:
            spectrum_id = self.database.add_spectrum(
                compound_name=spectrum_info.get('compound_name', 'Unknown'),
                spectrum_data=spectrum_info['spectrum_data'],
                chemical_formula=spectrum_info.get('chemical_formula', ''),
                cas_number=spectrum_info.get('cas_number', ''),
                measurement_conditions=spectrum_info.get('conditions', ''),
                metadata={
                    'source': spectrum_info.get('source', 'import'),
                    'original_filename': spectrum_info.get('original_filename', ''),
                    'import_timestamp': pd.Timestamp.now().isoformat(),
                    'is_preprocessed': spectrum_info.get('is_preprocessed', False)
                }
            )
            
            st.success(f"✅ Imported spectrum with ID: {spectrum_id}")
            return spectrum_id
            
        except Exception as e:
            st.error(f"❌ Import failed: {str(e)}")
            return None
    
    def _execute_batch_import(self, spectra_list: List[Dict], overwrite: bool, 
                            validate: bool, preprocess: bool, generate_features: bool):
        """Execute batch import with progress tracking."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        imported_count = 0
        errors = []
        
        for i, spectrum_info in enumerate(spectra_list):
            try:
                status_text.text(f"Importing {spectrum_info.get('compound_name', 'Unknown')}...")
                
                # Preprocess if requested
                if preprocess and not spectrum_info.get('is_preprocessed', False):
                    spectrum_info['spectrum_data'] = self.preprocessor.preprocess(
                        spectrum_info['spectrum_data']
                    )
                    spectrum_info['is_preprocessed'] = True
                
                # Import spectrum
                spectrum_id = self._import_single_spectrum(spectrum_info)
                
                if spectrum_id:
                    imported_count += 1
                
                progress_bar.progress((i + 1) / len(spectra_list))
                
            except Exception as e:
                errors.append(f"{spectrum_info.get('compound_name', 'Unknown')}: {str(e)}")
        
        # Final results
        status_text.text("Import complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Successfully Imported", imported_count)
        with col2:
            st.metric("Errors", len(errors))
        
        if errors:
            with st.expander("View Errors"):
                for error in errors:
                    st.text(f"❌ {error}")
        
        # Update database stats
        stats = self.database.get_database_stats()
        st.info(f"Database now contains {stats['total_spectra']} total spectra")


def create_import_templates():
    """Create downloadable import templates for users."""
    st.subheader("📋 Import Templates")
    
    st.write("Download these templates to see the expected data formats:")
    
    # CSV template
    csv_template = pd.DataFrame({
        'wavenumber': np.arange(200, 2000, 2),
        'intensity': np.random.random(900) * 1000
    })
    
    csv_buffer = io.StringIO()
    csv_template.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_buffer.getvalue(),
        file_name="spectrum_template.csv",
        mime="text/csv"
    )
    
    # JSON template
    json_template = {
        "compound_name": "Example Compound",
        "chemical_formula": "C6H6",
        "cas_number": "71-43-2",
        "spectrum": list(np.random.random(900) * 1000),
        "conditions": "Room temperature, 473nm laser",
        "source": "laboratory_measurement"
    }
    
    json_buffer = io.StringIO()
    json.dump(json_template, json_buffer, indent=2)
    
    st.download_button(
        label="📥 Download JSON Template",
        data=json_buffer.getvalue(),
        file_name="spectrum_template.json",
        mime="application/json"
    )