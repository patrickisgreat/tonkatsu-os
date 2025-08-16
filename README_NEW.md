# 🔬 Tonkatsu-OS: AI-Powered Raman Molecular Identification

A comprehensive, open-source Raman spectrometer control and AI-powered molecular identification platform. Built for DIY and repurposed spectrometers, Tonkatsu-OS combines advanced signal processing, machine learning, and intuitive interfaces to provide professional-grade molecular analysis capabilities.

## ✨ Key Features

### 🤖 AI-Powered Molecular Identification
- **Ensemble ML Models**: Random Forest, SVM, and Neural Network classifiers working together
- **Confidence Scoring**: Advanced uncertainty quantification and risk assessment
- **Peak Matching**: Characteristic Raman shift identification for molecular fingerprinting
- **Similarity Search**: Vectorized spectral database with cosine similarity matching

### 🔬 Advanced Signal Processing
- **Noise Reduction**: Cosmic ray removal and advanced filtering
- **Baseline Correction**: Asymmetric Least Squares (ALS) algorithm
- **Peak Detection**: Automated peak finding with property calculation
- **Multi-method Normalization**: Min-max, standard, L2, and area normalization

### 📊 Comprehensive Database System
- **SQLite Backend**: Efficient storage with vectorized spectral representations
- **Public Database Integration**: RRUFF and NIST database downloading and integration
- **Feature Extraction**: Automatic extraction of 50+ spectral features for ML
- **Metadata Management**: Chemical formulas, CAS numbers, measurement conditions

### 🖥️ Interactive Web Interface
- **Live Analysis**: Real-time spectrum acquisition and identification
- **Database Management**: Search, add, and visualize spectral data
- **Model Training**: Interactive ML model training and hyperparameter optimization
- **Data Visualization**: Advanced plotting and statistical analysis

## 🚀 Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/patrickisgreat/tonkatsu-os.git
cd tonkatsu-os
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run main.py
```

### 3. Get Started
1. **Demo Mode**: Use synthetic spectra to explore features without hardware
2. **Build Database**: Download RRUFF data or generate synthetic training data
3. **Train Models**: Use the ML training interface to build identification models
4. **Analyze Spectra**: Start identifying molecular samples!

## 🏗️ System Architecture

```
tonkatsu-os/
├── main.py                 # Streamlit web interface
├── database.py             # Spectral database management
├── preprocessing.py        # Signal processing pipeline
├── ml_models.py           # Machine learning classifiers
├── data_loader.py         # Public database integration
├── test_suite.py          # Comprehensive test framework
└── requirements.txt       # Dependencies
```

### Core Components

- **RamanSpectralDatabase**: SQLite-based storage with vectorized search
- **AdvancedPreprocessor**: Complete signal processing pipeline
- **EnsembleClassifier**: Multi-model ML system with confidence scoring
- **DataIntegrator**: Automated public database downloading and integration
- **PeakMatcher**: Characteristic peak identification algorithm

## 🧪 Scientific Capabilities

### Spectral Analysis
- **Wavelength Range**: Configurable (default: 2048 data points)
- **Peak Detection**: Automated with prominence and distance filtering
- **Feature Extraction**: 50+ statistical, spectral moment, and energy features
- **Quality Assessment**: Signal-to-noise ratio and spectral quality metrics

### Machine Learning
- **Ensemble Approach**: Combines RF, SVM, and Neural Network predictions
- **Cross-Validation**: Built-in model validation and performance assessment
- **Hyperparameter Optimization**: Grid search for optimal model parameters
- **Uncertainty Quantification**: Entropy-based confidence estimation

### Database Integration
- **RRUFF Database**: Automated downloading of mineral Raman spectra
- **NIST Integration**: Support for NIST spectral database formats
- **Synthetic Data**: Built-in generator for common organic compounds
- **Custom Data**: Easy import of user-generated spectral libraries

## 🔧 Hardware Compatibility

### Supported Spectrometers
- **B&W Tek Spectrometers**: Native support for 473nm systems
- **Serial Interface**: Standard RS-232/USB communication
- **Generic Support**: Configurable for other serial-based spectrometers

### Configuration
```python
SERIAL_PORT = "/dev/ttyUSB0"    # Adjust for your system
BAUD_RATE = 9600               # Standard B&W Tek rate
DATA_POINTS = 2048             # Spectral resolution
LASER_WAVELENGTH = 473.0       # nm
```

## 📊 Performance Metrics

### ML Model Performance
- **Accuracy**: Typically >85% on well-curated datasets
- **Confidence Calibration**: Isotonic regression for reliable probabilities
- **Processing Speed**: <1 second for complete analysis pipeline
- **Database Search**: Sub-second similarity search on 1000+ spectra

### Analysis Capabilities
- **Detection Limit**: Depends on sample and laser power
- **Spectral Range**: Full detector range supported
- **Peak Resolution**: Configurable (default: 5-10 cm⁻¹)
- **Noise Handling**: Robust cosmic ray and electronic noise removal

## 🧪 Example Workflows

### 1. Quick Identification
```python
# Acquire spectrum
spectrum = acquire_spectrum()

# Preprocess and analyze
processed = preprocessor.preprocess(spectrum)
features = preprocessor.spectral_features(processed)

# Identify molecule
predictions = classifier.predict([features])
confidence = confidence_scorer.calculate_confidence_score(...)

print(f"Identified: {predictions[0]['predicted_compound']}")
print(f"Confidence: {predictions[0]['confidence']:.1%}")
```

### 2. Database Building
```python
# Download public data
integrator = DataIntegrator(database)
results = integrator.download_and_integrate_rruff(max_spectra=100)

# Generate synthetic training data
synthetic_results = integrator.generate_and_integrate_synthetic()

print(f"Database now contains {database.get_database_stats()['total_spectra']} spectra")
```

### 3. Model Training
```python
# Train ensemble classifier
classifier = EnsembleClassifier()
results = classifier.train(X_features, y_labels)

# Save trained model
classifier.save_model("trained_model.pkl")

print(f"Model accuracy: {results['ensemble_accuracy']:.1%}")
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_suite.py
```

Tests include:
- Unit tests for all components
- Integration tests for workflows
- Performance benchmarks
- Database integrity checks

## 📊 Data Formats

### Spectral Data Storage
- **Raw Spectra**: Original intensity values
- **Preprocessed**: Cleaned and normalized data
- **Features**: Extracted characteristics for ML
- **Metadata**: Chemical formulas, conditions, sources

### Export Formats
- **CSV**: Standard format for data exchange
- **JSON**: Metadata and structured data
- **SQLite**: Complete database exports
- **Pickle**: Trained model serialization

## 🚡 Safety & Compliance

**⚠️ LASER SAFETY WARNING**: This system interfaces with Class 3B laser systems. Always:
- Wear appropriate laser safety eyewear (OD 4+ @ 473nm)
- Never view laser output directly
- Follow all local laser safety regulations
- Ensure proper beam containment and safety interlocks

## 🛠️ Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

### Architecture Principles
- **Modular Design**: Independent, testable components
- **Scientific Rigor**: Validated algorithms and statistical methods
- **User Experience**: Intuitive interfaces for complex operations
- **Extensibility**: Plugin architecture for new algorithms and databases

## 📚 Scientific Background

### Raman Spectroscopy
Raman spectroscopy is a powerful analytical technique that provides molecular fingerprints through inelastic light scattering. Each molecule has characteristic vibrational modes that appear as peaks in the Raman spectrum, enabling identification and quantification.

### Machine Learning Approach
Our ensemble approach combines:
- **Random Forest**: Handles non-linear relationships and feature importance
- **SVM**: Excellent for high-dimensional data with clear margins
- **Neural Networks**: Captures complex spectral patterns and interactions

## 🔗 References & Resources

- [RRUFF Database](https://rruff.info/): Comprehensive mineral Raman database
- [NIST Chemistry WebBook](https://webbook.nist.gov/): Chemical and spectroscopic data
- [Scikit-learn Documentation](https://scikit-learn.org/): Machine learning library
- [Streamlit Documentation](https://streamlit.io/): Web interface framework

## 📜 License

This project is open-source and available under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **@patrickisgreat**: Original creator and maintainer
- **LaserPointerForums Community**: Hardware hacking inspiration
- **RRUFF Project**: Public spectral database access
- **Open Science Community**: Making scientific tools accessible

---

**Built with ❤️ for the DIY science community**