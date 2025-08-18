# Tonkotsu-OS Raman Spectrometer Analysis System
## User Manual & Workflow Guide

### Table of Contents
1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Hardware Connection Workflow](#hardware-connection-workflow)
4. [Sample Analysis Workflow](#sample-analysis-workflow)
5. [Prediction & Identification Process](#prediction--identification-process)
6. [Machine Learning Training System](#machine-learning-training-system)
7. [Database Management](#database-management)
8. [Data Import & Export](#data-import--export)
9. [Troubleshooting](#troubleshooting)

---

## System Overview

The Tonkotsu-OS Raman Spectrometer Analysis System is a comprehensive platform for:
- **Real-time spectrometer control** and data acquisition
- **Advanced spectral analysis** with peak detection and preprocessing
- **Machine learning-powered compound identification** using ensemble classification
- **Comprehensive database management** with 30+ pharmaceutical compounds and mineral data
- **Beautiful spectral visualization** with interactive charts and peak identification

### Key Features
- Support for multiple spectrometer hardware interfaces
- Real-time and manual spectrum analysis modes
- Ensemble ML classifier for compound prediction
- Pharmaceutical database (180MB+ Springer Nature dataset)
- RRUFF mineral database integration
- Advanced spectral preprocessing and peak detection

---

## Getting Started

### Prerequisites
1. **Raman Spectrometer Hardware** (USB/Serial connection)
2. **System Running**: Backend API server and frontend interface
3. **Database Populated**: At least some reference spectra for comparison

### Initial Setup
1. **Start the Backend Server**:
   ```bash
   cd /path/to/tonkotsu-os
   python scripts/start_backend.py
   ```

2. **Launch Frontend Interface**:
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access Interface**: Navigate to `http://localhost:3000`

### External API Configuration (Optional)
To enable external database queries for unknown compounds:

1. **NIST Spectral Database**:
   ```bash
   export NIST_API_KEY="your_nist_api_key"
   ```

2. **ChemSpider Database**:
   ```bash
   export CHEMSPIDER_API_KEY="your_chemspider_key"
   ```

3. **Environment File** (recommended):
   Create `.env` file in project root:
   ```env
   NIST_API_KEY=your_nist_key_here
   CHEMSPIDER_API_KEY=your_chemspider_key_here
   ```

**Note**: External APIs are queried only when no local database matches are found.

---

## Hardware Connection Workflow

### Step 1: Connect Your Spectrometer
1. **Physical Connection**: Connect your Raman spectrometer via USB or serial port
2. **Driver Installation**: Ensure proper drivers are installed for your hardware
3. **Port Detection**: The system will auto-detect available COM/USB ports

### Step 2: Hardware Configuration
1. **Navigate** to the "Analyze" page in the interface
2. **Select Hardware Mode**: Choose "🔬 Hardware Mode" instead of "📝 Manual Input"
3. **Configure Port**: Select the correct serial port (e.g., /dev/ttyUSB0)
4. **Click Connect**: Connect to your spectrometer hardware
5. **Set Integration Time**: Adjust exposure time (50-5000ms) based on your sample

### Step 3: Hardware Testing
1. **Hardware Status**: Check the status panel shows "✅ Connected"
2. **Laser Status**: Verify laser status shows "ready"
3. **Test Acquisition**: Try capturing a spectrum with a known sample

---

## Sample Analysis Workflow

### Step 1: Sample Preparation
1. **Clean Sample**: Ensure sample surface is clean and free of contaminants
2. **Position Sample**: Place sample under the laser focal point
3. **Laser Safety**: Verify laser safety protocols are followed

### Step 2: Spectrum Acquisition
1. **Integration Time Setting**: Adjust integration time (50-5000ms) for optimal signal-to-noise ratio
2. **Capture Spectrum**: Click "Acquire & Analyze" to capture spectrum and automatically run analysis
3. **Automatic Analysis**: The system automatically analyzes the captured spectrum
4. **Review Results**: Check the analysis results and spectral chart

### Step 3: Analysis Configuration (Optional)
Before analyzing, you can configure:
- **Model Selection**: Choose which ML models to use (Random Forest, SVM, Neural Network, PLS)
- **Ensemble Method**: Select voting or weighted averaging
- **Preprocessing**: Enable/disable baseline correction and smoothing

---

## Prediction & Identification Process

The system uses a **4-tier analysis strategy** that prioritizes reliability:

### Step 1: Automatic Preprocessing
When you capture a spectrum, the system automatically:

1. **Preprocessing Pipeline**:
   - Baseline correction using asymmetric least squares
   - Savitzky-Golay smoothing for noise reduction
   - Normalization to unit area
   - Peak detection and characterization

2. **Feature Extraction**:
   - Peak positions and intensities
   - Spectral fingerprint regions
   - Statistical descriptors (centroid, moments, etc.)

### Step 2: Database Similarity Search (FIRST PRIORITY)
The system first searches your local database:

- **Cosine similarity** comparison against all 30+ known spectra
- **Correlation analysis** for pattern matching
- **Threshold**: 70% similarity required for confident match
- **Result**: If found, returns database compound with similarity score

### Step 3: External API Queries (SECOND PRIORITY)
If no database match found, queries external services:

- **NIST Spectral Database**: Government reference spectra
- **ChemSpider**: Royal Society of Chemistry database
- **API Configuration**: Set environment variables:
  ```bash
  export NIST_API_KEY="your_nist_key"
  export CHEMSPIDER_API_KEY="your_chemspider_key"
  ```
- **Timeout**: 15 seconds maximum per API

### Step 4: Machine Learning Classification (THIRD PRIORITY)
**Status**: ⚠️ **Currently mock implementation - needs real training**

The **Ensemble Classifier** combines four algorithms:
- **Random Forest**: 200 decision trees for robust predictions
- **Support Vector Machine**: RBF kernel for non-linear patterns  
- **Neural Network**: Multi-layer perceptron (100→50 neurons)
- **PLS Regression**: NIPALS algorithm optimized for spectroscopic data

**How ML Analysis Works**:
1. **Feature Extraction**: Extract ~67 features from each spectrum
2. **Preprocessing**: PCA dimensionality reduction (50 components)
3. **Ensemble Voting**: Each algorithm votes on compound identity
4. **Confidence Scoring**: Weighted combination of individual predictions

### Step 5: Fallback Analysis (LAST RESORT)
Simple rule-based identification:
- >10 peaks → "Benzene-like compound"
- High centroid → "Alcohol-like compound"  
- <5 peaks → "Simple molecule"

### Analysis Results Include:

1. **Primary Identification**:
   - Compound name and confidence score
   - **Method used**: database_similarity, external_api_nist, etc.
   - Chemical formula and CAS number (if available)

2. **Confidence Analysis**:
   - **Risk Level**: Low/Medium/High
   - **Components**: Similarity score, spectral quality, etc.
   - **Recommendation**: Guidance for interpretation

3. **Match Details**:
   - **Database matches**: Number found, similarity scores
   - **External API**: Response time, source database
   - **Fallback reason**: Why other methods failed

### Understanding Results:

**High Confidence (>80%)**:
- ✅ **Database match found** or **External API confirmed**
- ✅ **Risk Level: Low** - Results reliable

**Medium Confidence (60-80%)**:
- ⚠️ **Partial database match** or **External API uncertain**
- ⚠️ **Risk Level: Medium** - Consider additional validation

**Low Confidence (<60%)**:
- ❌ **No database matches**, **APIs failed** 
- ❌ **Risk Level: High** - Manual verification required

---

## Machine Learning Training System

### Overview - How ML Training Works

**Machine Learning** allows the system to recognize patterns in spectral data and predict compounds even when exact database matches aren't available. Think of it as teaching the computer to "learn chemistry" from your reference spectra.

### Current Implementation Status

**⚠️ IMPORTANT**: The training system currently provides **mock results only**. Here's what exists vs. what needs implementation:

#### ✅ What's Implemented:
- **Training Interface**: Complete UI with progress tracking
- **Training API**: Backend endpoints for configuration
- **ML Architecture**: Full ensemble classifier code structure
- **Mock Training**: Simulates 10-second training with fake results
- **Model Configuration**: PCA components, validation split, hyperparameter optimization

#### ❌ What's Missing (Critical Gap):
- **Real Training Logic**: No actual model training occurs
- **Data Pipeline**: Database spectra aren't loaded for training
- **Feature Engineering**: Spectral features aren't extracted for ML
- **Model Persistence**: Trained models aren't saved/loaded
- **Production Predictions**: ML classifier isn't used in analysis

### How Real ML Training Should Work

#### Step 1: Data Preparation
```
Database Spectra (30+ compounds) 
    ↓
Feature Extraction (67 features per spectrum)
    ↓
Train/Validation Split (80%/20%)
    ↓
Preprocessing (normalization, PCA)
```

#### Step 2: Ensemble Training
**Random Forest** (200 trees):
- Learns which spectral features distinguish compounds
- Handles noise and variations in spectra
- Provides feature importance rankings

**Support Vector Machine** (RBF kernel):
- Finds optimal boundaries between compound classes
- Excellent for high-dimensional spectral data
- Handles non-linear relationships

**Neural Network** (100→50 neurons):
- Learns complex spectral patterns
- Adapts to subtle chemical signatures
- Provides probability distributions

**PLS Regression** (NIPALS algorithm):
- Specifically designed for spectroscopic data
- Handles multicollinearity in spectral features
- Excellent for chemical analysis

#### Step 3: Model Evaluation
- **Cross-Validation**: 5-fold validation for robust metrics
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Shows which compounds are confused
- **Feature Importance**: Identifies most discriminative spectral regions

### Using the Training Interface

#### Prerequisites for Real Training:
1. **Minimum 10 spectra** per compound class
2. **Diverse spectral conditions** (different integration times, conditions)
3. **Quality reference data** with proper peak identification

#### Training Configuration:

**PCA Components (10-200)**:
- Reduces spectral dimensionality while preserving information
- **50 components**: Good default for most applications
- **Higher values**: More detail, longer training time
- **Lower values**: Faster training, may lose information

**Validation Split (0.1-0.4)**:
- **0.2 (20%)**: Standard split for model validation
- **Higher split**: More validation data, less training data
- **Lower split**: More training data, less validation confidence

**Hyperparameter Optimization**:
- ✅ **Enabled**: Grid search for optimal parameters (slower)
- ❌ **Disabled**: Use default parameters (faster)

#### Training Process:
1. **Navigate to Training page**
2. **Configure parameters** based on your dataset size
3. **Click "Start Training"** (currently shows mock 10-second progress)
4. **Monitor progress** with real-time updates
5. **Review metrics** after completion

### Understanding Training Results

#### Model Performance Metrics:

**Ensemble Accuracy (Target: >85%)**:
- Overall performance combining all algorithms
- Higher is better, but >90% may indicate overfitting

**Individual Model Performance**:
- **Random Forest**: Usually most robust
- **SVM**: Often highest accuracy on clean data
- **Neural Network**: Best for complex patterns
- **PLS**: Most reliable for spectroscopic data

**Feature Importance**:
- Shows which spectral regions are most discriminative
- Helps validate chemical knowledge
- Guides future data collection

#### Training Troubleshooting:

**Low Accuracy (<70%)**:
- ✅ Add more training spectra per compound
- ✅ Check data quality (noise, artifacts)
- ✅ Ensure proper preprocessing
- ✅ Verify compound labels are correct

**Overfitting (Validation << Training accuracy)**:
- ✅ Reduce model complexity
- ✅ Add more diverse training data
- ✅ Increase regularization parameters

**Long Training Times**:
- ❌ Reduce PCA components
- ❌ Disable hyperparameter optimization
- ❌ Use smaller validation split

### Integration with Analysis Workflow

Once properly trained, the ML system would:

1. **Automatic Integration**: Analysis API uses trained model when database matches fail
2. **Confidence Scoring**: ML predictions include uncertainty estimates
3. **Fallback Priority**: Database → External APIs → **Trained ML** → Simple rules
4. **Continuous Learning**: Retrain periodically as database grows

### Implementation Roadmap

**Phase 1: Basic Training (High Priority)**
- ✅ Load spectra from database
- ✅ Extract spectral features (peaks, centroid, etc.)
- ✅ Train ensemble models on real data
- ✅ Save/load trained models

**Phase 2: Advanced Features**
- ✅ Cross-validation and metrics
- ✅ Feature importance analysis
- ✅ Hyperparameter optimization
- ✅ Model comparison tools

**Phase 3: Production Integration**
- ✅ Real-time ML predictions in analysis
- ✅ Confidence calibration
- ✅ Active learning (learn from user feedback)
- ✅ Automated retraining

---

## Database Management

### Current Database Contents
- **32+ Pharmaceutical Compounds**: Including APIs like acetaminophen, aspirin, caffeine
- **50+ Mineral Spectra**: RRUFF database entries with high-quality references
- **Synthetic Test Data**: Generated spectra for algorithm validation

### Adding New Reference Spectra

1. **Manual Entry**: Analyze known samples and save to database
2. **Batch Import**: Use "Import" page to load spectral databases
3. **Quality Control**: Verify all imports with proper metadata

### Database Search & Browse
- **Search Functionality**: Find compounds by name, formula, or CAS number
- **Filter Options**: Sort by minerals, organics, pharmaceuticals
- **Metadata Viewing**: Access full spectral parameters and conditions

---

## Data Import & Export

### Supported Import Formats
- **CSV**: Wavenumber, intensity pairs
- **TXT**: Space/tab-delimited spectral data
- **JSON**: Structured spectral metadata
- **Manufacturer Formats**: Various proprietary formats

### Database Downloads
1. **RRUFF Raman Database**: High-quality mineral spectra
2. **Pharmaceutical Database**: Springer Nature API dataset  
3. **RRUFF Chemistry**: Microprobe analysis data
4. **RRUFF Infrared**: Complementary IR spectroscopy

### Export Options
- **Individual Spectra**: Download as CSV/JSON
- **Analysis Reports**: PDF reports with predictions
- **Database Backup**: Full database export

---

## Troubleshooting

### Hardware Issues
**Spectrometer Not Detected**:
- Check USB/serial connections
- Verify driver installation
- Try different USB ports
- Check device manager for errors

**Poor Signal Quality**:
- Increase integration time
- Adjust laser focus
- Clean sample and optics
- Check for ambient light interference

### Software Issues
**Analysis Fails**:
- Verify spectrum has adequate signal
- Check database connection
- Ensure sufficient reference data
- Review error logs in backend

**Prediction Confidence Low**:
- Improve spectrum quality
- Add more reference spectra
- Check for sample contamination
- Verify measurement conditions

### Performance Optimization
**Slow Analysis**:
- Reduce spectral resolution if acceptable
- Limit database search scope
- Close unnecessary applications
- Ensure adequate RAM available

---

## Advanced Features

### Mixture Analysis
- **Multi-component Detection**: Identify mixture components
- **Concentration Estimation**: Quantitative analysis capabilities
- **Spectral Deconvolution**: Separate overlapping peaks

### Custom Model Training
- **Retrain Classifier**: Add your own spectral data
- **Domain Adaptation**: Optimize for specific sample types
- **Validation Metrics**: Track model performance

### Batch Processing
- **Multiple Sample Analysis**: Process series of samples
- **Automated Reporting**: Generate batch reports
- **Statistical Analysis**: Compare sample populations

---

## Best Practices

### Measurement Quality
1. **Consistent Conditions**: Maintain temperature, humidity
2. **Laser Stability**: Allow warmup time
3. **Sample Handling**: Minimize contamination
4. **Reference Standards**: Regular calibration checks

### Data Management
1. **Metadata Recording**: Document all measurement conditions
2. **Version Control**: Track database updates
3. **Backup Strategy**: Regular database backups
4. **Quality Assurance**: Validate new entries

### Safety Considerations
1. **Laser Safety**: Proper eye protection
2. **Sample Handling**: Chemical safety protocols
3. **Electrical Safety**: Proper grounding
4. **Documentation**: Maintain safety logs

---

## Support & Resources

### Technical Support
- **Error Logs**: Check browser console and backend logs
- **Documentation**: Refer to API documentation
- **Community**: GitHub issues and discussions

### Further Reading
- **Raman Spectroscopy Theory**: Understanding spectral interpretation
- **Machine Learning**: Algorithm details and optimization
- **Database Design**: Spectral database best practices

---

## Current Implementation Status

### ✅ Fully Implemented Features
- **Hardware Mode**: Connect to spectrometers via serial/USB ports
- **Manual Input Mode**: Analyze spectrum data by pasting values
- **Database Search**: Browse and search 30+ pharmaceutical compounds and minerals
- **Spectral Visualization**: Beautiful interactive charts with peak detection
- **Machine Learning Analysis**: Ensemble classifier with multiple algorithms
- **Database Management**: Import RRUFF and pharmaceutical databases
- **Analysis Configuration**: Configurable ML models and preprocessing

### ⚠️ Current Limitations
- **No Live Preview**: Cannot preview spectrum before acquisition
- **Limited Hardware Support**: Generic serial communication only
- **No Real-time Parameters**: Cannot adjust laser power during acquisition
- **Single Shot Mode**: No continuous scanning or averaging modes
- **Manual Quality Assessment**: No automated spectrum quality metrics

### 🚧 Planned Features
- **Live spectral preview** with real-time display
- **Advanced hardware control** for laser power and scanning parameters
- **Automated quality assessment** and acquisition optimization
- **Multi-scan averaging** and background subtraction
- **Calibration routines** for wavelength and intensity
- **Batch processing** for multiple samples

---

*This manual covers the current implementation. The system provides a solid foundation for Raman spectroscopy analysis with room for hardware-specific enhancements. For specific technical details, refer to the API documentation and source code comments.*