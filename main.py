# raman_ml_pipeline/main.py

import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pybaselines import als
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import streamlit as st
import os

SERIAL_PORT = "/dev/ttyUSB0"  # Change as needed
BAUD_RATE = 9600
DATA_POINTS = 2048
EXPORT_PATH = "exported_spectra"
os.makedirs(EXPORT_PATH, exist_ok=True)


def acquire_spectrum(port=SERIAL_PORT, baudrate=BAUD_RATE):
    with serial.Serial(port, baudrate, timeout=2) as ser:
        ser.write(b'a\r\n')  # ASCII mode
        ser.write(b'I200\r\n')  # 200ms integration
        ser.write(b'S\r\n')
        raw_data = ser.read_until(expected=b'\r\n')
        spectrum = np.array([int(x) for x in raw_data.decode(errors='ignore').split()[:DATA_POINTS]])
        return spectrum


def preprocess_spectrum(spectrum):
    smoothed = savgol_filter(spectrum, 11, 3)
    baseline_corrected = smoothed - als(smoothed)[0]
    normalized = (baseline_corrected - np.min(baseline_corrected)) / (np.max(baseline_corrected) - np.min(baseline_corrected))
    return normalized


def load_training_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y


def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pca = PCA(n_components=20)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_pca, y_train)
    preds = clf.predict(X_test_pca)
    print(classification_report(y_test, preds))
    return clf, pca


def export_spectrum(spectrum, label="unknown"):
    filename = os.path.join(EXPORT_PATH, f"spectrum_{label}.csv")
    np.savetxt(filename, spectrum, delimiter=",", header="intensity", comments="")
    print(f"Spectrum exported to {filename}")


def main():
    st.title("Raman Spectrometer Interface")

    if st.button("Acquire Spectrum"):
        spectrum = acquire_spectrum()
        processed = preprocess_spectrum(spectrum)

        st.line_chart(processed)
        export_spectrum(processed)

        if st.checkbox("Run ML Classification"):
            if os.path.exists("labeled_spectra.csv"):
                X, y = load_training_data('labeled_spectra.csv')
                clf, pca = train_classifier(X, y)
                pred = clf.predict(pca.transform([processed]))
                st.success(f"Predicted class: {pred[0]}")
            else:
                st.warning("labeled_spectra.csv not found.")


if __name__ == '__main__':
    main()
