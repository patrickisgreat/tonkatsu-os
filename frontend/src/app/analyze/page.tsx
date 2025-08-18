'use client'

import { useState, useEffect } from 'react'
import { api } from '@/utils/api'
import { SpectralChart } from '@/components/SpectralChart'

interface HardwareStatus {
  connected: boolean;
  port?: string;
  laser_status?: string;
  temperature?: number;
  last_communication?: string;
}

interface Port {
  device: string;
  description: string;
  hwid: string;
}

export default function AnalyzePage() {
  const [spectrumData, setSpectrumData] = useState<string>('')
  const [analyzedSpectrumData, setAnalyzedSpectrumData] = useState<number[]>([])
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [hardwareStatus, setHardwareStatus] = useState<HardwareStatus | null>(null)
  const [availablePorts, setAvailablePorts] = useState<Port[]>([])
  const [selectedPort, setSelectedPort] = useState<string>('/dev/ttyUSB0')
  const [integrationTime, setIntegrationTime] = useState<number>(200)
  const [mode, setMode] = useState<'hardware' | 'manual'>('hardware')
  const [connecting, setConnecting] = useState(false)
  const [acquiring, setAcquiring] = useState(false)
  
  // Analysis configuration
  const [analysisConfig, setAnalysisConfig] = useState({
    models: {
      random_forest: true,
      svm: true,
      neural_network: true,
      pls_regression: false
    },
    pls_config: {
      n_components: 10,
      max_iter: 500,
      tolerance: 1e-6
    },
    ensemble_method: 'voting', // 'voting' or 'weighted'
    preprocessing: true
  })

  // Load hardware status and available ports on mount
  useEffect(() => {
    loadHardwareStatus()
    scanForPorts()
  }, [])

  const loadHardwareStatus = async () => {
    try {
      const status = await api.getHardwareStatus()
      setHardwareStatus(status)
    } catch (error) {
      console.error('Failed to load hardware status:', error)
    }
  }

  const scanForPorts = async () => {
    try {
      const ports = await api.scanPorts()
      setAvailablePorts(ports)
      if (ports.length > 0 && !selectedPort) {
        setSelectedPort(ports[0].device)
      }
    } catch (error) {
      console.error('Failed to scan ports:', error)
    }
  }

  const handleConnect = async () => {
    setConnecting(true)
    try {
      const response = await api.connectHardware(selectedPort)
      if (response.success) {
        await loadHardwareStatus()
        alert('Connected to Raman spectrometer successfully!')
      } else {
        alert(`Connection failed: ${response.message}`)
      }
    } catch (error) {
      console.error('Connection failed:', error)
      alert('Connection failed. Check if the device is connected and the port is correct.')
    } finally {
      setConnecting(false)
    }
  }

  const handleDisconnect = async () => {
    try {
      await api.disconnectHardware()
      await loadHardwareStatus()
      alert('Disconnected from spectrometer')
    } catch (error) {
      console.error('Disconnect failed:', error)
      alert('Disconnect failed')
    }
  }

  const handleAcquireSpectrum = async () => {
    setAcquiring(true)
    try {
      const spectrumArray = await api.acquireSpectrum(integrationTime)
      
      // Convert to comma-separated string for display and analysis
      setSpectrumData(spectrumArray.join(', '))
      
      // Automatically analyze the acquired spectrum
      await analyzeSpectrum(spectrumArray)
      
    } catch (error) {
      console.error('Spectrum acquisition failed:', error)
      alert('Failed to acquire spectrum. Check hardware connection.')
    } finally {
      setAcquiring(false)
    }
  }

  const analyzeSpectrum = async (data: number[]) => {
    setLoading(true)
    try {
      // Store the spectrum data for visualization
      setAnalyzedSpectrumData(data)
      
      // Include analysis configuration in the request
      const analysisResult = await api.analyzeSpectrum(data, analysisConfig)
      setResult(analysisResult)
    } catch (error) {
      console.error('Analysis failed:', error)
      alert('Analysis failed. Please check your spectrum data.')
    } finally {
      setLoading(false)
    }
  }

  const handleManualAnalyze = async () => {
    if (!spectrumData.trim()) return
    
    try {
      // Parse comma-separated values
      const data = spectrumData.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
      
      if (data.length === 0) {
        alert('Please enter valid spectrum data (comma-separated numbers)')
        return
      }
      
      await analyzeSpectrum(data)
    } catch (error) {
      console.error('Manual analysis failed:', error)
      alert('Analysis failed. Please check your input.')
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Live Raman Analysis</h1>
        <p className="text-gray-600 mt-2">Connect your spectrometer for real-time molecular identification</p>
      </div>

      {/* Mode Selection */}
      <div className="mb-6">
        <div className="flex space-x-4">
          <button
            onClick={() => setMode('hardware')}
            className={`px-4 py-2 rounded-md font-medium ${
              mode === 'hardware' 
                ? 'bg-primary-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            🔬 Hardware Mode
          </button>
          <button
            onClick={() => setMode('manual')}
            className={`px-4 py-2 rounded-md font-medium ${
              mode === 'manual' 
                ? 'bg-primary-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            📝 Manual Input
          </button>
        </div>
      </div>

      {/* Analysis Configuration */}
      <div className="mb-6">
        <div className="card">
          <div className="card-header">
            <h3 className="text-md font-medium">⚙️ Analysis Configuration</h3>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Model Selection */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Select Models</h4>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={analysisConfig.models.random_forest}
                      onChange={(e) => setAnalysisConfig({
                        ...analysisConfig,
                        models: { ...analysisConfig.models, random_forest: e.target.checked }
                      })}
                      className="mr-2 rounded"
                    />
                    <span className="text-sm">Random Forest</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={analysisConfig.models.svm}
                      onChange={(e) => setAnalysisConfig({
                        ...analysisConfig,
                        models: { ...analysisConfig.models, svm: e.target.checked }
                      })}
                      className="mr-2 rounded"
                    />
                    <span className="text-sm">Support Vector Machine</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={analysisConfig.models.neural_network}
                      onChange={(e) => setAnalysisConfig({
                        ...analysisConfig,
                        models: { ...analysisConfig.models, neural_network: e.target.checked }
                      })}
                      className="mr-2 rounded"
                    />
                    <span className="text-sm">Neural Network</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={analysisConfig.models.pls_regression}
                      onChange={(e) => setAnalysisConfig({
                        ...analysisConfig,
                        models: { ...analysisConfig.models, pls_regression: e.target.checked }
                      })}
                      className="mr-2 rounded"
                    />
                    <span className="text-sm">PLS Regression</span>
                  </label>
                </div>
              </div>

              {/* PLS Configuration */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">PLS Configuration</h4>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Components
                    </label>
                    <input
                      type="number"
                      value={analysisConfig.pls_config.n_components}
                      onChange={(e) => setAnalysisConfig({
                        ...analysisConfig,
                        pls_config: { ...analysisConfig.pls_config, n_components: parseInt(e.target.value) }
                      })}
                      min="1"
                      max="50"
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500"
                      disabled={!analysisConfig.models.pls_regression}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Max Iterations
                    </label>
                    <input
                      type="number"
                      value={analysisConfig.pls_config.max_iter}
                      onChange={(e) => setAnalysisConfig({
                        ...analysisConfig,
                        pls_config: { ...analysisConfig.pls_config, max_iter: parseInt(e.target.value) }
                      })}
                      min="100"
                      max="2000"
                      step="100"
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500"
                      disabled={!analysisConfig.models.pls_regression}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Ensemble Method
                    </label>
                    <select
                      value={analysisConfig.ensemble_method}
                      onChange={(e) => setAnalysisConfig({
                        ...analysisConfig,
                        ensemble_method: e.target.value as 'voting' | 'weighted'
                      })}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500"
                    >
                      <option value="voting">Majority Voting</option>
                      <option value="weighted">Weighted Average</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={analysisConfig.preprocessing}
                  onChange={(e) => setAnalysisConfig({
                    ...analysisConfig,
                    preprocessing: e.target.checked
                  })}
                  className="mr-2 rounded"
                />
                <span className="text-sm font-medium">Enable preprocessing (baseline correction, smoothing)</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Section */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">
              {mode === 'hardware' ? '🔬 Hardware Control' : '📝 Manual Input'}
            </h2>
          </div>
          <div className="card-content space-y-4">
            {mode === 'hardware' ? (
              <>
                {/* Hardware Status */}
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-medium text-gray-900 mb-2">Hardware Status</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Connection:</span>
                      <span className={`font-medium ${
                        hardwareStatus?.connected ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {hardwareStatus?.connected ? '✅ Connected' : '❌ Disconnected'}
                      </span>
                    </div>
                    {hardwareStatus?.connected && (
                      <>
                        <div className="flex justify-between">
                          <span>Port:</span>
                          <span className="font-mono text-xs">{hardwareStatus.port}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Laser:</span>
                          <span className={`font-medium ${
                            hardwareStatus.laser_status === 'ready' ? 'text-green-600' : 'text-yellow-600'
                          }`}>
                            {hardwareStatus.laser_status}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Connection Controls */}
                {!hardwareStatus?.connected && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Serial Port
                    </label>
                    <div className="flex space-x-2">
                      <select
                        value={selectedPort}
                        onChange={(e) => setSelectedPort(e.target.value)}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                      >
                        <option value="/dev/ttyUSB0">/dev/ttyUSB0 (Default)</option>
                        {availablePorts.map((port) => (
                          <option key={port.device} value={port.device}>
                            {port.device} - {port.description}
                          </option>
                        ))}
                      </select>
                      <button
                        onClick={scanForPorts}
                        className="px-3 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50"
                        title="Scan for ports"
                      >
                        🔄
                      </button>
                    </div>
                    
                    <button
                      onClick={handleConnect}
                      disabled={connecting}
                      className="btn-primary w-full mt-4 disabled:opacity-50"
                    >
                      {connecting ? 'Connecting...' : 'Connect to Spectrometer'}
                    </button>
                  </div>
                )}

                {/* Acquisition Controls */}
                {hardwareStatus?.connected && (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Integration Time (ms)
                      </label>
                      <input
                        type="number"
                        value={integrationTime}
                        onChange={(e) => setIntegrationTime(parseInt(e.target.value))}
                        min="50"
                        max="5000"
                        step="50"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Higher values = better signal quality, longer acquisition time
                      </p>
                    </div>

                    <button
                      onClick={handleAcquireSpectrum}
                      disabled={acquiring || loading}
                      className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {acquiring ? '🔬 Acquiring Spectrum...' : '🔬 Acquire & Analyze'}
                    </button>

                    <button
                      onClick={handleDisconnect}
                      className="w-full px-4 py-2 text-red-600 border border-red-300 rounded-md hover:bg-red-50"
                    >
                      Disconnect
                    </button>
                  </div>
                )}

                {/* Spectrum Preview */}
                {spectrumData && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Acquired Spectrum (first 10 points)
                    </label>
                    <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                      {spectrumData.split(',').slice(0, 10).join(', ')}...
                    </div>
                  </div>
                )}
              </>
            ) : (
              <>
                {/* Manual Input Mode */}
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
                  onClick={handleManualAnalyze}
                  disabled={loading || !spectrumData.trim()}
                  className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Analyzing...' : 'Analyze Spectrum'}
                </button>
              </>
            )}
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

                {/* Spectral Chart */}
                {analyzedSpectrumData.length > 0 && (
                  <div className="bg-white border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-4">Analyzed Spectrum</h4>
                    <SpectralChart
                      spectrumData={analyzedSpectrumData}
                      compoundName={result.predicted_compound}
                      showPeaks={true}
                      height={400}
                      color="rgb(34, 197, 94)"
                      backgroundColor="rgba(34, 197, 94, 0.05)"
                    />
                  </div>
                )}
                
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