'use client'

import { useState } from 'react'
import { api } from '@/utils/api'

export default function AnalyzePage() {
  const [spectrumData, setSpectrumData] = useState<string>('')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const handleAnalyze = async () => {
    if (!spectrumData.trim()) return
    
    setLoading(true)
    try {
      // Parse comma-separated values
      const data = spectrumData.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
      
      if (data.length === 0) {
        alert('Please enter valid spectrum data (comma-separated numbers)')
        return
      }
      
      const analysisResult = await api.analyzeSpectrum({ spectrum_data: data, preprocess: true })
      setResult(analysisResult)
    } catch (error) {
      console.error('Analysis failed:', error)
      alert('Analysis failed. Please check your input.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Spectrum Analysis</h1>
        <p className="text-gray-600 mt-2">Analyze Raman spectra with AI-powered molecular identification</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Section */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Input Spectrum Data</h2>
          </div>
          <div className="card-content space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Spectrum Data (comma-separated values)
              </label>
              <textarea
                value={spectrumData}
                onChange={(e) => setSpectrumData(e.target.value)}
                placeholder="100, 200, 300, 150, 50..."
                className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
            
            <button
              onClick={handleAnalyze}
              disabled={loading || !spectrumData.trim()}
              className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing...' : 'Analyze Spectrum'}
            </button>
          </div>
        </div>

        {/* Results Section */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Analysis Results</h2>
          </div>
          <div className="card-content">
            {result ? (
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-green-800">
                    {result.predicted_compound}
                  </h3>
                  <p className="text-green-600">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Individual Model Predictions:</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Random Forest:</span>
                      <span>{result.individual_predictions?.random_forest?.compound} ({(result.individual_predictions?.random_forest?.confidence * 100).toFixed(1)}%)</span>
                    </div>
                    <div className="flex justify-between">
                      <span>SVM:</span>
                      <span>{result.individual_predictions?.svm?.compound} ({(result.individual_predictions?.svm?.confidence * 100).toFixed(1)}%)</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Neural Network:</span>
                      <span>{result.individual_predictions?.neural_network?.compound} ({(result.individual_predictions?.neural_network?.confidence * 100).toFixed(1)}%)</span>
                    </div>
                    {result.individual_predictions?.pls_regression && (
                      <div className="flex justify-between">
                        <span>PLS Regression:</span>
                        <span>{result.individual_predictions.pls_regression.compound} ({(result.individual_predictions.pls_regression.confidence * 100).toFixed(1)}%)</span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Confidence Analysis:</h4>
                  <div className="text-sm space-y-1">
                    <div className="flex justify-between">
                      <span>Risk Level:</span>
                      <span className={`font-medium ${
                        result.confidence_analysis?.risk_level === 'low' ? 'text-green-600' :
                        result.confidence_analysis?.risk_level === 'medium' ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {result.confidence_analysis?.risk_level?.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-gray-600 mt-2">
                      {result.confidence_analysis?.recommendation}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>Enter spectrum data and click "Analyze" to see results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}