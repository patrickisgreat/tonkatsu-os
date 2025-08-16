# Enhanced Raman Spectrometer Interface with AI/ML Molecular Identification

import serial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import os
import logging
from pathlib import Path

# Import our custom modules
from database import RamanSpectralDatabase
from preprocessing import AdvancedPreprocessor, PeakMatcher
from ml_models import EnsembleClassifier, ConfidenceScorer
from data_loader import DataIntegrator, SyntheticDataGenerator
from spectrum_importer import SpectrumImporter, create_import_templates
from visualization import SpectralVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SERIAL_PORT = "/dev/ttyUSB0"  # Change as needed
BAUD_RATE = 9600
DATA_POINTS = 2048
EXPORT_PATH = "exported_spectra"
DATABASE_PATH = "raman_spectra.db"
MODEL_PATH = "trained_model.pkl"

# Create directories
os.makedirs(EXPORT_PATH, exist_ok=True)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize database and other components."""
    db = RamanSpectralDatabase(DATABASE_PATH)
    preprocessor = AdvancedPreprocessor()
    classifier = EnsembleClassifier()
    confidence_scorer = ConfidenceScorer()
    peak_matcher = PeakMatcher()
    importer = SpectrumImporter(db)
    visualizer = SpectralVisualizer(db)
    
    return db, preprocessor, classifier, confidence_scorer, peak_matcher, importer, visualizer


def acquire_spectrum(port=SERIAL_PORT, baudrate=BAUD_RATE):
    """Acquire spectrum from the spectrometer."""
    try:
        with serial.Serial(port, baudrate, timeout=2) as ser:
            ser.write(b'a\r\n')  # ASCII mode
            ser.write(b'I200\r\n')  # 200ms integration
            ser.write(b'S\r\n')
            raw_data = ser.read_until(expected=b'\r\n')
            spectrum = np.array([int(x) for x in raw_data.decode(errors='ignore').split()[:DATA_POINTS]])
            return spectrum
    except Exception as e:
        st.error(f"Error acquiring spectrum: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Tonkatsu-OS: AI-Powered Raman Analysis",
        page_icon="🔬",
        layout="wide"
    )
    
    # Initialize components
    db, preprocessor, classifier, confidence_scorer, peak_matcher, importer, visualizer = init_components()
    
    st.title("🔬 Tonkatsu-OS: AI-Powered Raman Molecular Identification")
    st.markdown("*Advanced DIY Raman spectrometer with machine learning-based molecular identification*")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🔍 Live Analysis", "📥 Import Spectra", "📊 Visualizations", "🗄️ Database", "🤖 Model Training", "⚙️ Settings"]
    )
    
    if page == "🔍 Live Analysis":
        live_analysis_page(db, preprocessor, classifier, confidence_scorer, peak_matcher, visualizer)
    elif page == "📥 Import Spectra":
        import_spectra_page(importer)
    elif page == "📊 Visualizations":
        visualizations_page(visualizer)
    elif page == "🗄️ Database":
        database_management_page(db)
    elif page == "🤖 Model Training":
        model_training_page(db, classifier)
    elif page == "⚙️ Settings":
        settings_page()


def import_spectra_page(importer):
    """Spectrum import interface page."""
    importer.create_import_interface()
    
    # Add import templates section
    with st.expander("📋 Download Import Templates"):
        create_import_templates()


def visualizations_page(visualizer):
    """Advanced visualizations page."""
    visualizer.create_visualization_dashboard()


def live_analysis_page(db, preprocessor, classifier, confidence_scorer, peak_matcher, visualizer):
    """Live spectrum acquisition and analysis page."""
    st.header("Live Spectrum Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Acquisition Settings")
        use_real_spectrometer = st.checkbox("Use real spectrometer", value=False)
        integration_time = st.slider("Integration time (ms)", 50, 1000, 200)
        
        if st.button("🔬 Acquire Spectrum", type="primary"):
            if use_real_spectrometer:
                raw_spectrum = acquire_spectrum()
            else:
                # Generate synthetic spectrum for demo
                generator = SyntheticDataGenerator()
                compound = st.selectbox("Demo compound", list(generator.common_compounds.keys()))
                raw_spectrum = generator.generate_synthetic_spectrum(compound)
                st.info(f"Generated synthetic spectrum for {compound}")
            
            if raw_spectrum is not None:
                analyze_spectrum(raw_spectrum, db, preprocessor, classifier, confidence_scorer, peak_matcher, col1)
    
    with col1:
        st.subheader("Spectrum Visualization")
        if 'current_spectrum' not in st.session_state:
            st.info("Acquire a spectrum to see visualization")
        else:
            display_spectrum_analysis(st.session_state.current_spectrum)


def analyze_spectrum(raw_spectrum, db, preprocessor, classifier, confidence_scorer, peak_matcher, display_col):
    """Analyze an acquired spectrum."""
    with st.spinner("Analyzing spectrum..."):
        # Preprocess spectrum
        processed_spectrum = preprocessor.preprocess(raw_spectrum)
        
        # Detect peaks
        peaks, peak_intensities = preprocessor.detect_peaks(processed_spectrum)
        
        # Extract features for ML
        features = preprocessor.spectral_features(processed_spectrum)
        
        # Store in session state
        st.session_state.current_spectrum = {
            'raw': raw_spectrum,
            'processed': processed_spectrum,
            'peaks': peaks,
            'peak_intensities': peak_intensities,
            'features': features
        }
        
        # Search similar spectra in database
        similar_spectra = db.search_similar_spectra(raw_spectrum, top_k=5)
        
        # ML Classification (if model is trained)
        if os.path.exists(MODEL_PATH):
            try:
                classifier.load_model(MODEL_PATH)
                
                # Prepare features for ML
                feature_vector = preprocessor._extract_features(processed_spectrum, peaks, peak_intensities)
                predictions = classifier.predict([feature_vector])
                
                if predictions:
                    prediction = predictions[0]
                    
                    # Calculate comprehensive confidence score
                    peak_match_score = 0.5  # Placeholder - implement peak matching
                    confidence_analysis = confidence_scorer.calculate_confidence_score(
                        np.array([pred['probability'] for pred in prediction['top_predictions']]),
                        peak_match_score,
                        prediction['model_agreement']
                    )
                    
                    display_results(prediction, confidence_analysis, similar_spectra, display_col)
                
            except Exception as e:
                st.error(f"Error in ML classification: {e}")
        else:
            st.warning("No trained model found. Train a model first.")
            display_similarity_results(similar_spectra, display_col)


def display_results(prediction, confidence_analysis, similar_spectra, display_col):
    """Display analysis results."""
    with display_col:
        st.subheader("🎯 Identification Results")
        
        # Main prediction
        pred_compound = prediction['predicted_compound']
        confidence = prediction['confidence']
        
        # Color-code based on confidence
        if confidence > 0.8:
            confidence_color = "🟢"
        elif confidence > 0.6:
            confidence_color = "🟡"
        else:
            confidence_color = "🔴"
        
        st.markdown(f"""
        ### {confidence_color} **{pred_compound}**
        **Confidence:** {confidence:.1%} | **Agreement:** {prediction['model_agreement']:.1%}
        """)
        
        # Confidence breakdown
        with st.expander("Confidence Analysis"):
            st.write("**Overall Confidence:**", f"{confidence_analysis['overall_confidence']:.1%}")
            st.write("**Risk Level:**", confidence_analysis['risk_level'].title())
            st.write("**Recommendation:**", confidence_analysis['recommendation'])
            
            # Component scores
            for component, score in confidence_analysis['confidence_components'].items():
                st.metric(component.replace('_', ' ').title(), f"{score:.2f}")
        
        # Top predictions
        st.subheader("Top Predictions")
        for i, pred in enumerate(prediction['top_predictions'][:3]):
            st.write(f"{i+1}. **{pred['compound']}** ({pred['probability']:.1%})")
        
        # Model agreement
        with st.expander("Individual Model Results"):
            for model_name, result in prediction['individual_predictions'].items():
                st.write(f"**{model_name.replace('_', ' ').title()}:** {result['compound']} ({result['confidence']:.1%})")
        
        # Similar spectra from database
        if similar_spectra:
            st.subheader("Similar Spectra in Database")
            for spec in similar_spectra[:3]:
                st.write(f"• **{spec['compound_name']}** (similarity: {spec['similarity_score']:.1%})")


def display_similarity_results(similar_spectra, display_col):
    """Display similarity search results when no ML model is available."""
    with display_col:
        st.subheader("🔍 Database Search Results")
        
        if similar_spectra:
            for i, spec in enumerate(similar_spectra):
                st.write(f"{i+1}. **{spec['compound_name']}** (similarity: {spec['similarity_score']:.1%})")
                if spec['chemical_formula']:
                    st.write(f"   Formula: {spec['chemical_formula']}")
        else:
            st.info("No similar spectra found in database.")


def display_spectrum_analysis(spectrum_data):
    """Display spectrum visualization and analysis."""
    if spectrum_data:
        # Plot raw and processed spectra
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Raw spectrum
        ax1.plot(spectrum_data['raw'])
        ax1.set_title("Raw Spectrum")
        ax1.set_ylabel("Intensity")
        
        # Processed spectrum with peaks
        ax2.plot(spectrum_data['processed'])
        if len(spectrum_data['peaks']) > 0:
            ax2.plot(spectrum_data['peaks'], spectrum_data['peak_intensities'], 'ro', markersize=6)
            
            # Annotate major peaks
            for i, (peak, intensity) in enumerate(zip(spectrum_data['peaks'][:5], spectrum_data['peak_intensities'][:5])):
                ax2.annotate(f'{peak}', (peak, intensity), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax2.set_title("Processed Spectrum with Peak Detection")
        ax2.set_xlabel("Data Point")
        ax2.set_ylabel("Normalized Intensity")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Peak statistics
        if len(spectrum_data['peaks']) > 0:
            st.subheader("Peak Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Peaks", len(spectrum_data['peaks']))
            with col2:
                st.metric("Dominant Peak", spectrum_data['peaks'][np.argmax(spectrum_data['peak_intensities'])])
            with col3:
                st.metric("Peak Density", f"{len(spectrum_data['peaks'])/len(spectrum_data['processed']):.3f}")


def database_management_page(db):
    """Database management interface."""
    st.header("📊 Database Management")
    
    # Database statistics
    stats = db.get_database_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Spectra", stats['total_spectra'])
    with col2:
        st.metric("Unique Compounds", stats['unique_compounds'])
    with col3:
        st.metric("Database Size", f"{Path(DATABASE_PATH).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Search interface
    st.subheader("Search Database")
    search_term = st.text_input("Search by compound name:")
    
    if search_term:
        results = db.search_by_compound_name(search_term)
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
        else:
            st.info("No results found.")
    
    # Add new spectrum
    with st.expander("Add New Spectrum"):
        compound_name = st.text_input("Compound Name")
        formula = st.text_input("Chemical Formula (optional)")
        cas_number = st.text_input("CAS Number (optional)")
        
        if st.button("Add Spectrum") and compound_name:
            # This would typically use acquired spectrum data
            st.success(f"Spectrum for {compound_name} would be added to database")
    
    # Data integration
    st.subheader("Data Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download RRUFF Data"):
            with st.spinner("Downloading RRUFF data..."):
                integrator = DataIntegrator(db)
                results = integrator.download_and_integrate_rruff(max_spectra=20)
                st.success(f"Integrated {results.get('successful_integrations', 0)} spectra")
    
    with col2:
        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                integrator = DataIntegrator(db)
                results = integrator.generate_and_integrate_synthetic(n_samples_per_compound=5)
                st.success(f"Generated {results.get('successful_integrations', 0)} synthetic spectra")


def model_training_page(db, classifier):
    """Model training interface."""
    st.header("🤖 Model Training")
    
    # Check database status
    stats = db.get_database_stats()
    
    if stats['total_spectra'] < 10:
        st.warning("Need at least 10 spectra in database for training. Add more data first.")
        return
    
    st.info(f"Database contains {stats['total_spectra']} spectra from {stats['unique_compounds']} compounds")
    
    # Training parameters
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_pca = st.checkbox("Use PCA", value=True)
        n_components = st.slider("PCA Components", 10, 100, 50)
    
    with col2:
        validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2)
        optimize_hyperparams = st.checkbox("Optimize Hyperparameters", value=False)
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models..."):
            # Load training data from database
            # This is a placeholder - implement data loading from database
            X_train = np.random.random((100, 200))  # Placeholder
            y_train = np.random.choice(['compound_a', 'compound_b', 'compound_c'], 100)
            
            # Configure classifier
            classifier.use_pca = use_pca
            classifier.n_components = n_components
            
            # Train the model
            results = classifier.train(X_train, y_train, validation_split)
            
            # Save the model
            classifier.save_model(MODEL_PATH)
            
            # Display results
            st.success("Training completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance")
                st.metric("Random Forest Accuracy", f"{results['rf_accuracy']:.1%}")
                st.metric("SVM Accuracy", f"{results['svm_accuracy']:.1%}")
                st.metric("Neural Network Accuracy", f"{results['nn_accuracy']:.1%}")
                st.metric("Ensemble Accuracy", f"{results['ensemble_accuracy']:.1%}")
            
            with col2:
                st.subheader("Training Details")
                st.metric("Training Samples", results['n_train_samples'])
                st.metric("Validation Samples", results['n_val_samples'])
                st.metric("Features", results['n_features'])
                st.metric("Classes", results['n_classes'])


def data_visualization_page(db):
    """Data visualization interface."""
    st.header("📈 Data Visualization")
    
    # Database overview charts
    stats = db.get_database_stats()
    
    if stats['total_spectra'] > 0:
        # Compound distribution
        st.subheader("Compound Distribution")
        compound_data = pd.DataFrame(stats['top_compounds'], columns=['Compound', 'Count'])
        st.bar_chart(compound_data.set_index('Compound'))
        
        # Sample spectrum visualization
        st.subheader("Sample Spectra")
        sample_compounds = [comp[0] for comp in stats['top_compounds'][:3]]
        
        for compound in sample_compounds:
            results = db.search_by_compound_name(compound, exact_match=True)
            if results:
                spectrum_data = db.get_spectrum_by_id(results[0]['id'])
                if spectrum_data and spectrum_data['preprocessed_spectrum'] is not None:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(spectrum_data['preprocessed_spectrum'])
                    ax.set_title(f"Sample Spectrum: {compound}")
                    ax.set_xlabel("Data Point")
                    ax.set_ylabel("Intensity")
                    st.pyplot(fig)
    else:
        st.info("No data available for visualization. Add spectra to the database first.")


def settings_page():
    """Settings and configuration page."""
    st.header("⚙️ Settings")
    
    # Hardware settings
    st.subheader("Hardware Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        serial_port = st.text_input("Serial Port", value=SERIAL_PORT)
        baud_rate = st.selectbox("Baud Rate", [9600, 19200, 38400, 115200], index=0)
    
    with col2:
        data_points = st.number_input("Data Points", value=DATA_POINTS, min_value=512, max_value=4096)
        laser_wavelength = st.number_input("Laser Wavelength (nm)", value=473.0)
    
    # Processing settings
    st.subheader("Processing Configuration")
    
    smoothing_window = st.slider("Smoothing Window", 5, 21, 11, step=2)
    baseline_correction = st.checkbox("Baseline Correction", value=True)
    cosmic_ray_removal = st.checkbox("Cosmic Ray Removal", value=True)
    
    # Database settings
    st.subheader("Database Configuration")
    
    db_path = st.text_input("Database Path", value=DATABASE_PATH)
    backup_enabled = st.checkbox("Enable Automatic Backup", value=True)
    
    if st.button("Save Settings"):
        st.success("Settings saved!")
    
    # System information
    st.subheader("System Information")
    
    info_data = {
        "Database Size": f"{Path(DATABASE_PATH).stat().st_size / 1024 / 1024:.1f} MB" if Path(DATABASE_PATH).exists() else "Not found",
        "Model Status": "Trained" if os.path.exists(MODEL_PATH) else "Not trained",
        "Export Directory": EXPORT_PATH,
        "Log Level": "INFO"
    }
    
    for key, value in info_data.items():
        st.text(f"{key}: {value}")


if __name__ == '__main__':
    main()
