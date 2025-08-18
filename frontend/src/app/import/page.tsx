'use client'

import { useState } from 'react'
import { api } from '@/utils/api'

export default function ImportPage() {
  const [dragActive, setDragActive] = useState(false)
  const [importStatus, setImportStatus] = useState<string>('')

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = async (file: File) => {
    setImportStatus('Uploading file...')
    try {
      await api.uploadFile(file, '/import/spectrum')
      setImportStatus('✅ File imported successfully!')
    } catch (error) {
      console.error('Import failed:', error)
      setImportStatus('❌ Import failed. Please try again.')
    }
  }

  const downloadRRUFF = async () => {
    setImportStatus('Downloading RRUFF Raman database...')
    try {
      await api.downloadRRUFFData(50)
      setImportStatus('✅ RRUFF Raman database downloaded successfully!')
    } catch (error) {
      console.error('RRUFF download failed:', error)
      setImportStatus('❌ RRUFF download failed. Please try again.')
    }
  }

  const downloadRRUFFChemistry = async () => {
    setImportStatus('Downloading RRUFF chemistry database...')
    try {
      await api.downloadRRUFFChemistryData(25)
      setImportStatus('✅ RRUFF chemistry database downloaded successfully!')
    } catch (error) {
      console.error('RRUFF chemistry download failed:', error)
      setImportStatus('❌ RRUFF chemistry download failed. Please try again.')
    }
  }

  const downloadRRUFFInfrared = async () => {
    setImportStatus('Downloading RRUFF infrared database...')
    try {
      await api.downloadRRUFFInfraredData(25)
      setImportStatus('✅ RRUFF infrared database downloaded successfully!')
    } catch (error) {
      console.error('RRUFF infrared download failed:', error)
      setImportStatus('❌ RRUFF infrared download failed. Please try again.')
    }
  }

  const downloadPharmaceuticalData = async () => {
    setImportStatus('Downloading pharmaceutical database...')
    try {
      await api.downloadPharmaceuticalData(50)
      setImportStatus('✅ Pharmaceutical database downloaded successfully!')
    } catch (error) {
      console.error('Pharmaceutical download failed:', error)
      setImportStatus('❌ Pharmaceutical download failed. Please try again.')
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Import Spectral Data</h1>
        <p className="text-gray-600 mt-2">Upload spectrum files or download public databases</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* File Upload */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Upload Spectrum Files</h2>
          </div>
          <div className="card-content">
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive 
                  ? 'border-primary-500 bg-primary-50' 
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="space-y-4">
                <div className="mx-auto h-12 w-12 text-gray-400">
                  <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm text-gray-600">
                    Drag and drop files here, or{" "}
                    <label className="text-primary-600 hover:text-primary-700 cursor-pointer">
                      browse
                      <input
                        type="file"
                        className="hidden"
                        accept=".csv,.txt,.json"
                        onChange={handleFileInput}
                      />
                    </label>
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Supported formats: CSV, TXT, JSON, XLSX
                  </p>
                </div>
              </div>
            </div>
            
            {importStatus && (
              <div className="mt-4 p-3 bg-gray-50 border rounded text-sm">
                {importStatus}
              </div>
            )}
          </div>
        </div>

        {/* Public Databases */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Public Databases</h2>
          </div>
          <div className="card-content space-y-4">
            <div className="border rounded-lg p-4">
              <h3 className="font-medium text-gray-900">RRUFF Raman Spectra</h3>
              <p className="text-sm text-gray-600 mt-1">
                High-quality mineral Raman spectra from the RRUFF project (excellent oriented)
              </p>
              <button
                onClick={downloadRRUFF}
                className="btn-primary mt-3"
              >
                Download Raman Spectra (50)
              </button>
            </div>

            <div className="border rounded-lg p-4">
              <h3 className="font-medium text-gray-900">RRUFF Chemistry Data</h3>
              <p className="text-sm text-gray-600 mt-1">
                Microprobe chemistry analysis data for mineral identification
              </p>
              <button
                onClick={downloadRRUFFChemistry}
                className="btn-secondary mt-3"
              >
                Download Chemistry Data (25)
              </button>
            </div>

            <div className="border rounded-lg p-4">
              <h3 className="font-medium text-gray-900">RRUFF Infrared Spectra</h3>
              <p className="text-sm text-gray-600 mt-1">
                Processed infrared spectroscopy data for mineral analysis
              </p>
              <button
                onClick={downloadRRUFFInfrared}
                className="btn-secondary mt-3"
              >
                Download IR Spectra (25)
              </button>
            </div>
            
            <div className="border rounded-lg p-4">
              <h3 className="font-medium text-gray-900">Pharmaceutical Database</h3>
              <p className="text-sm text-gray-600 mt-1">
                Raman spectra of active pharmaceutical ingredients (APIs) from Springer Nature
              </p>
              <button
                onClick={downloadPharmaceuticalData}
                className="btn-primary mt-3"
              >
                Download Pharma Spectra (50)
              </button>
            </div>
            
            <div className="border rounded-lg p-4 opacity-60">
              <h3 className="font-medium text-gray-900">NIST Database</h3>
              <p className="text-sm text-gray-600 mt-1">
                Coming soon - NIST spectral database integration
              </p>
              <button
                disabled
                className="btn-secondary mt-3 opacity-50 cursor-not-allowed"
              >
                Coming Soon
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Import History */}
      <div className="mt-8">
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Recent Imports</h2>
          </div>
          <div className="card-content">
            <div className="text-center py-8 text-gray-500">
              <p>No recent imports</p>
              <p className="text-sm mt-1">Imported files will appear here</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}