# tonkatsu-os
A RAMAN Spectroscopy data pipeline. 
aTonkatsu-OS

A DIY, open-source Raman spectrometer control and machine learning platform. Built for repurposed B&W Tek 473nm spectrometers (like the Pharmanex S1), this tool lets you capture, visualize, export, and classify Raman spectra using a friendly web interface and powerful backend.

🔧 Features

Serial interface to B&W Tek spectrometers via pySerial

Signal preprocessing: smoothing, baseline correction, normalization

Interactive web UI (Streamlit) for:

Acquiring spectra

Live visualization

ML-based classification

Exports to CSV for open-data reuse

Scikit-learn integration for training and inference

PCA dimensionality reduction

Easily extendable with new models or preprocessing methods

🚀 Quick Start

1. Clone the repo

git clone git@github.com:patrickisgreat/tonkatsu-os.git
cd tonkatsu-os

2. Install dependencies

pip install -r requirements.txt

3. Run the app

streamlit run main.py

🐳 Docker Support

Build and run with Docker:

docker build -t tonkatsu-os .
docker run --device=/dev/ttyUSB0 -p 8501:8501 tonkatsu-os

Make sure to replace /dev/ttyUSB0 with your serial device path.

🧒 Sample Training Data Format

To enable ML classification, place a CSV in the project root named labeled_spectra.csv:

label,0,1,2,3,...,2047
chlorophyll,0.12,0.14,0.16,...,0.21
aspirin,0.34,0.32,0.30,...,0.27

label column: the name of the compound or material

Remaining columns: 2048-point normalized intensity spectrum

✅ Tests

Run unit tests with:

pytest test_preprocess.py

📁 Exported Data

Spectra are saved to the exported_spectra/ directory in CSV format after acquisition.

🚡 Safety Warning

This project interfaces with a high-powered 473 nm DPSS laser. Ensure proper laser safety practices and protective eyewear are used at all times. Never view laser output directly.

📚 Credits

Built by @patrickisgreatInspired by the LaserPointerForums and open hardware hacking community.

🧠 Future Ideas

Auto peak detection and annotation

Raman shift conversion (cm⁻¹)

REST API for remote integration

TensorFlow-powered compound classification