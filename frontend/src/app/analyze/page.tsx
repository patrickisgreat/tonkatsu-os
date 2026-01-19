'use client'

import { useState, useEffect } from 'react'
import type { KeyboardEvent } from 'react'
import { api } from '@/utils/api'
import { SpectralChart } from '@/components/SpectralChart'
import { SpectrumPreview } from '@/components/SpectrumPreview'
import type {
  HardwareStatus,
  AcquisitionResponse,
  ReferenceSpectrum,
  CalibrationDetail,
  CalibrationSummary,
} from '@/types/spectrum'

interface Port {
  device: string;
  description: string;
  hwid: string;
}

export default function AnalyzePage() {
  const [spectrumData, setSpectrumData] = useState<string>('')
  const [rawSpectrumData, setRawSpectrumData] = useState<number[]>([])
  const [processedSpectrumData, setProcessedSpectrumData] = useState<number[]>([])
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [hardwareStatus, setHardwareStatus] = useState<HardwareStatus | null>(null)
  const [availablePorts, setAvailablePorts] = useState<Port[]>([])
  const [selectedPort, setSelectedPort] = useState<string>('/dev/ttyUSB0')
  const [integrationTime, setIntegrationTime] = useState<number>(200)
  const [mode, setMode] = useState<'hardware' | 'manual'>('hardware')
  const [connecting, setConnecting] = useState(false)
  const [acquiring, setAcquiring] = useState(false)
  const [acquisitionInfo, setAcquisitionInfo] = useState<AcquisitionResponse | null>(null)
  const [simulateMode, setSimulateMode] = useState(false)
  const [simulationFile, setSimulationFile] = useState<string>('')
  const [hintInput, setHintInput] = useState('')
  const [hints, setHints] = useState<string[]>([])
  const [referenceSpectra, setReferenceSpectra] = useState<ReferenceSpectrum[]>([])
  const [referenceLoading, setReferenceLoading] = useState(false)
  const [referenceError, setReferenceError] = useState<string | null>(null)

  // Dark subtraction state
  const [hasDark, setHasDark] = useState(false)
  const [darkInfo, setDarkInfo] = useState<{mean?: number; data_points?: number} | null>(null)
  const [acquiringDark, setAcquiringDark] = useState(false)
  const [useDarkSubtraction, setUseDarkSubtraction] = useState(true)
  const [averageCount, setAverageCount] = useState(1)

  // Calibration state
  const [instrumentId, setInstrumentId] = useState('default')
  const [calibrationList, setCalibrationList] = useState<CalibrationSummary[]>([])
  const [activeCalibration, setActiveCalibration] = useState<CalibrationDetail | null>(null)
  const [calibrationName, setCalibrationName] = useState('')
  const [calibrationNotes, setCalibrationNotes] = useState('')
  const [calibrationLaser, setCalibrationLaser] = useState('')
  const [selectedCalibrationId, setSelectedCalibrationId] = useState<number | ''>('')
  const [pixelA, setPixelA] = useState('')
  const [wavenumberA, setWavenumberA] = useState('')
  const [pixelB, setPixelB] = useState('')
  const [wavenumberB, setWavenumberB] = useState('')
  const [calibrationError, setCalibrationError] = useState<string | null>(null)
  const [calibrationBusy, setCalibrationBusy] = useState(false)

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
    void loadHardwareStatus()
    void scanForPorts()

    const interval = setInterval(() => {
      void loadHardwareStatus()
    }, 5000)

    return () => {
      clearInterval(interval)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!instrumentId) return
    void loadCalibrations(instrumentId)
    void loadActiveCalibration(instrumentId)
  }, [instrumentId])

  const loadHardwareStatus = async () => {
    try {
      const status = await api.getHardwareStatus()
      setHardwareStatus(status)
      if (status.connected) {
        setSimulateMode(Boolean(status.simulate))
        const inferredId = status.port || selectedPort || 'default'
        setInstrumentId((prev) => (prev === 'default' ? inferredId : prev))
        await loadDarkInfo()
      }
    } catch (error) {
      console.error('Failed to load hardware status:', error)
    }
  }

  const formatTimestamp = (value?: string | null) => {
    if (!value) return '—'
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) {
      return value
    }
    return date.toLocaleString()
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

  const handleSimulatorToggle = (checked: boolean) => {
    setSimulateMode(checked)
    if (checked) {
      setSelectedPort('simulator')
    } else if (selectedPort === 'simulator') {
      setSelectedPort('/dev/ttyUSB0')
    }
  }

  const handleConnect = async () => {
    setConnecting(true)
    try {
      const response = await api.connectHardware(
        simulateMode
          ? {
              simulate: true,
              simulationFile: simulationFile || undefined,
              port: selectedPort
            }
          : { port: selectedPort }
      )
      if (response.success) {
        await loadHardwareStatus()
        alert(response.message ?? 'Spectrometer connection established')
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
      setAcquisitionInfo(null)
      setRawSpectrumData([])
      setProcessedSpectrumData([])
      alert('Disconnected from spectrometer')
    } catch (error) {
      console.error('Disconnect failed:', error)
      alert('Disconnect failed')
    }
  }

  // Dark spectrum handlers
  const handleAcquireDark = async () => {
    if (!confirm('Block the laser beam or remove sample, then click OK to acquire dark spectrum.')) {
      return
    }
    setAcquiringDark(true)
    try {
      const data = await api.acquireDark({
        integrationTime,
        averages: averageCount,
      })
      setHasDark(true)
      setDarkInfo({
        mean: data.data.reduce((a: number, b: number) => a + b, 0) / data.data.length,
        data_points: data.data.length
      })
      alert('Dark spectrum acquired! Future acquisitions will be corrected.')
    } catch (error: any) {
      console.error('Dark acquisition failed:', error)
      alert('Failed to acquire dark spectrum')
    } finally {
      setAcquiringDark(false)
    }
  }

  const handleClearDark = async () => {
    try {
      await api.clearDark()
      setHasDark(false)
      setDarkInfo(null)
    } catch (error) {
      console.error('Failed to clear dark:', error)
    }
  }

  const loadDarkInfo = async () => {
    try {
      const info = await api.getDarkInfo()
      setHasDark(info.has_dark)
      if (info.has_dark) {
        setDarkInfo({ mean: info.mean, data_points: info.data_points })
      }
    } catch (error) {
      console.error('Failed to load dark info:', error)
    }
  }

  const loadCalibrations = async (targetInstrumentId: string) => {
    try {
      const calibrations = await api.listCalibrations(targetInstrumentId)
      setCalibrationList(calibrations)
    } catch (error) {
      console.error('Failed to load calibrations:', error)
    }
  }

  const loadActiveCalibration = async (targetInstrumentId: string) => {
    try {
      const calibration = await api.getActiveCalibration(targetInstrumentId)
      setActiveCalibration(calibration)
    } catch (error) {
      setActiveCalibration(null)
    }
  }

  const handleActivateCalibration = async (calibrationId: number) => {
    try {
      await api.activateCalibration(calibrationId, instrumentId)
      await loadCalibrations(instrumentId)
      await loadActiveCalibration(instrumentId)
    } catch (error) {
      console.error('Failed to activate calibration:', error)
      alert('Failed to activate calibration')
    }
  }

  const handleCreateCalibration = async () => {
    const name = calibrationName.trim()
    if (!name) {
      setCalibrationError('Enter a calibration name.')
      return
    }

    const pixelAValue = Number(pixelA)
    const pixelBValue = Number(pixelB)
    const wavenumberAValue = Number(wavenumberA)
    const wavenumberBValue = Number(wavenumberB)
    const axisLength = rawSpectrumData.length || hardwareStatus?.data_points || 2048

    if (![pixelAValue, pixelBValue, wavenumberAValue, wavenumberBValue].every(Number.isFinite)) {
      setCalibrationError('Enter two pixel indices and two wavenumber values.')
      return
    }
    if (pixelAValue === pixelBValue) {
      setCalibrationError('Pixel indices must be different.')
      return
    }
    if (pixelAValue < 0 || pixelBValue < 0 || pixelAValue >= axisLength || pixelBValue >= axisLength) {
      setCalibrationError(`Pixel indices must be within 0-${axisLength - 1}.`)
      return
    }

    const slope = (wavenumberBValue - wavenumberAValue) / (pixelBValue - pixelAValue)
    const axisData = Array.from({ length: axisLength }, (_, i) => wavenumberAValue + slope * (i - pixelAValue))
    const laserValue = calibrationLaser.trim() ? Number(calibrationLaser) : undefined

    if (calibrationLaser.trim() && !Number.isFinite(laserValue)) {
      setCalibrationError('Laser wavelength must be a number.')
      return
    }

    setCalibrationError(null)
    setCalibrationBusy(true)
    try {
      await api.createCalibration({
        name,
        instrument_id: instrumentId,
        axis_data: axisData,
        laser_wavelength: laserValue,
        notes: calibrationNotes.trim() || undefined,
        set_active: true,
      })
      setCalibrationName('')
      setCalibrationNotes('')
      setCalibrationLaser('')
      setPixelA('')
      setPixelB('')
      setWavenumberA('')
      setWavenumberB('')
      await loadCalibrations(instrumentId)
      await loadActiveCalibration(instrumentId)
    } catch (error) {
      console.error('Failed to create calibration:', error)
      setCalibrationError('Failed to save calibration.')
    } finally {
      setCalibrationBusy(false)
    }
  }

  const handleAcquireSpectrum = async () => {
    setAcquiring(true)
    try {
      setProcessedSpectrumData([])

      // Use corrected endpoint if dark subtraction is enabled and dark is available
      const endpoint = useDarkSubtraction && hasDark ? 'corrected' : 'acquire'
      const acquisition = await (endpoint === 'corrected'
        ? api.acquireSpectrumCorrected({
            integrationTime,
            simulate: simulateMode,
            simulationFile: simulationFile || undefined,
            averages: averageCount,
          })
        : api.acquireSpectrum({
            integrationTime,
            simulate: simulateMode,
            simulationFile: simulationFile || undefined,
            averages: averageCount,
          }))

      setAcquisitionInfo(acquisition)
      const spectrumArray = Array.isArray(acquisition?.data) ? acquisition.data : null
      if (!spectrumArray) {
        throw new Error('Failed to read spectrum data from response.')
      }

      // Convert to comma-separated string for display and analysis
      setSpectrumData(spectrumArray.join(', '))

      // Automatically analyze the acquired spectrum
      await analyzeSpectrum(spectrumArray)
      await loadHardwareStatus()
    } catch (error: any) {
      console.error('Spectrum acquisition failed:', error)
      const message = error?.message ?? 'Failed to acquire spectrum. Check hardware connection.'
      alert(message)
    } finally {
      setAcquiring(false)
    }
  }

  const analyzeSpectrum = async (data: number[]) => {
    setLoading(true)
    try {
      if (!Array.isArray(data)) {
        throw new Error('Spectrum data is missing or invalid.')
      }
      setRawSpectrumData(data)

      let processedData = data
      if (analysisConfig.preprocessing) {
        try {
          processedData = await api.preprocessSpectrum(data)
        } catch (error) {
          console.error('Preprocessing failed:', error)
          processedData = data
        }
      }
      setProcessedSpectrumData(processedData)

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
      
      setAcquisitionInfo(null)
      setProcessedSpectrumData([])
      await analyzeSpectrum(data)
    } catch (error) {
      console.error('Manual analysis failed:', error)
      alert('Analysis failed. Please check your input.')
    }
  }

  const handleAddHint = () => {
    const value = hintInput.trim()
    if (!value) return
    if (!hints.includes(value)) {
      setHints((prev) => [...prev, value])
    }
    setHintInput('')
    setReferenceError(null)
    setReferenceSpectra([])
  }

  const handleHintKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      handleAddHint()
    }
  }

  const handleRemoveHint = (hint: string) => {
    setHints((prev) => {
      const next = prev.filter((item) => item !== hint)
      if (next.length === 0) {
        setReferenceSpectra([])
        setReferenceError(null)
      }
      return next
    })
  }

  const handleFetchReferences = async () => {
    if (hints.length === 0) {
      setReferenceError('Add at least one hint to fetch reference spectra.')
      return
    }

    setReferenceLoading(true)
    setReferenceError(null)
    setReferenceSpectra([])
    try {
      const spectra = await api.fetchReferenceSpectra(hints)
      setReferenceSpectra(spectra)
      if (!spectra.length) {
        setReferenceError('No reference spectra returned yet. Integration is in progress.')
      }
    } catch (error) {
      console.error('Reference lookup failed:', error)
      setReferenceError('Failed to fetch reference spectra. Please try again later.')
    } finally {
      setReferenceLoading(false)
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

            <div className="mt-6 border-t pt-4">
              <h4 className="text-md font-medium text-gray-900 mb-2">Reference Hints (beta)</h4>
              <p className="text-sm text-gray-600 mb-3">
                Suggest suspected compounds or functional groups to target external reference libraries.
              </p>
              <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-2 space-y-2 sm:space-y-0">
                <input
                  type="text"
                  value={hintInput}
                  onChange={(e) => setHintInput(e.target.value)}
                  onKeyDown={handleHintKeyDown}
                  placeholder="e.g. aromatic ring, caffeine"
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <button
                  onClick={handleAddHint}
                  className="btn-secondary whitespace-nowrap"
                >
                  Add Hint
                </button>
              </div>

              {hints.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3">
                  {hints.map((hint) => (
                    <span
                      key={hint}
                      className="inline-flex items-center rounded-full bg-primary-50 px-3 py-1 text-xs text-primary-700"
                    >
                      {hint}
                      <button
                        onClick={() => handleRemoveHint(hint)}
                        className="ml-2 text-primary-500 hover:text-primary-700"
                        aria-label={`Remove ${hint}`}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}

              <button
                onClick={handleFetchReferences}
                className="btn-secondary mt-4"
                disabled={referenceLoading}
              >
                {referenceLoading ? 'Fetching references...' : 'Fetch Reference Spectra'}
              </button>

              {referenceError && (
                <p className="mt-2 text-xs text-red-600">{referenceError}</p>
              )}

              {!referenceError && !referenceLoading && referenceSpectra.length === 0 && hints.length > 0 && (
                <p className="mt-2 text-xs text-gray-500">
                  No reference spectra found yet. Adjust hints or try fetching again.
                </p>
              )}
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
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-gray-900">Hardware Status</h3>
                    <button
                      onClick={() => void loadHardwareStatus()}
                      className="text-xs text-primary-600 hover:text-primary-700 font-medium"
                    >
                      Refresh
                    </button>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Connection:</span>
                      <span className={`font-medium ${hardwareStatus?.connected ? 'text-green-600' : 'text-red-600'}`}>
                        {hardwareStatus?.connected ? '✅ Connected' : '❌ Disconnected'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Mode:</span>
                      <span className="font-medium">
                        {hardwareStatus?.simulate ? 'Simulator' : 'Hardware'}
                      </span>
                    </div>
                    {hardwareStatus?.port && (
                      <div className="flex justify-between">
                        <span>Port:</span>
                        <span className="font-mono text-xs">{hardwareStatus.port}</span>
                      </div>
                    )}
                    {typeof hardwareStatus?.temperature === 'number' && (
                      <div className="flex justify-between">
                        <span>Temperature:</span>
                        <span>{hardwareStatus.temperature?.toFixed(1)}°C</span>
                      </div>
                    )}
                    {hardwareStatus?.connected && (
                      <div className="flex justify-between">
                        <span>Laser:</span>
                        <span className={`font-medium ${hardwareStatus.laser_status === 'ready' ? 'text-green-600' : 'text-yellow-600'}`}>
                          {hardwareStatus.laser_status}
                        </span>
                      </div>
                    )}
                    {hardwareStatus?.data_points && (
                      <div className="flex justify-between">
                        <span>Data Points:</span>
                        <span>{hardwareStatus.data_points}</span>
                      </div>
                    )}
                    {hardwareStatus?.last_source && (
                      <div className="flex justify-between">
                        <span>Last Source:</span>
                        <span className="font-medium capitalize">{hardwareStatus.last_source}</span>
                      </div>
                    )}
                    {hardwareStatus?.simulate && hardwareStatus.simulation_file && (
                      <div className="flex justify-between">
                        <span>Simulation File:</span>
                        <span className="text-xs truncate max-w-[60%]">{hardwareStatus.simulation_file}</span>
                      </div>
                    )}
                    {hardwareStatus?.last_acquired_at && (
                      <div className="flex justify-between">
                        <span>Last Acquisition:</span>
                        <span className="text-xs">{formatTimestamp(hardwareStatus.last_acquired_at)}</span>
                      </div>
                    )}
                    {hardwareStatus?.last_communication && (
                      <div className="flex justify-between">
                        <span>Last Communication:</span>
                        <span className="text-xs">{formatTimestamp(hardwareStatus.last_communication)}</span>
                      </div>
                    )}
                    {hardwareStatus?.last_error && (
                      <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
                        <span className="font-semibold">Last error:</span> {hardwareStatus.last_error}
                      </div>
                    )}
                  </div>
                </div>

                {/* Connection Controls */}
                {!hardwareStatus?.connected && (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="flex items-center text-sm">
                        <input
                          type="checkbox"
                          checked={simulateMode}
                          onChange={(e) => handleSimulatorToggle(e.target.checked)}
                          className="mr-2 rounded"
                        />
                        <span>Use hardware simulator (development)</span>
                      </label>
                      {simulateMode && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Simulation File (optional)
                          </label>
                          <input
                            type="text"
                            value={simulationFile}
                            onChange={(e) => setSimulationFile(e.target.value)}
                            placeholder="e.g. data/simulations/spectrum.csv"
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                          />
                          <p className="text-xs text-gray-500 mt-1">
                            Provide a CSV or JSON path to replay recorded spectra.
                          </p>
                        </div>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Serial Port
                      </label>
                      <div className="flex space-x-2">
                        <select
                          value={selectedPort}
                          onChange={(e) => setSelectedPort(e.target.value)}
                          disabled={simulateMode}
                          className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-60 disabled:cursor-not-allowed"
                        >
                          {(simulateMode || selectedPort === 'simulator') && (
                            <option value="simulator">Simulator (virtual)</option>
                          )}
                          <option value="/dev/ttyUSB0">/dev/ttyUSB0 (Default)</option>
                          {availablePorts.map((port) => (
                            <option key={port.device} value={port.device}>
                              {port.device} - {port.description}
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={scanForPorts}
                          disabled={simulateMode}
                          className="px-3 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                          title="Scan for ports"
                        >
                          🔄
                        </button>
                      </div>
                      {simulateMode && (
                        <p className="text-xs text-gray-500 mt-1">
                          Port selection is disabled while the simulator is active.
                        </p>
                      )}
                    </div>
                    
                    <button
                      onClick={handleConnect}
                      disabled={connecting}
                      className="btn-primary w-full disabled:opacity-50"
                    >
                      {connecting
                        ? simulateMode
                          ? 'Starting simulator...'
                          : 'Connecting...'
                        : simulateMode
                          ? 'Start Simulator'
                          : 'Connect to Spectrometer'}
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
                        onChange={(e) => {
                          const value = parseInt(e.target.value, 10)
                          setIntegrationTime(Number.isFinite(value) ? value : 200)
                        }}
                        min="50"
                        max="5000"
                        step="50"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Higher values = better signal quality, longer acquisition time
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Averages (scans)
                      </label>
                      <input
                        type="number"
                        value={averageCount}
                        onChange={(e) => {
                          const value = parseInt(e.target.value, 10)
                          setAverageCount(Number.isFinite(value) ? Math.max(1, value) : 1)
                        }}
                        min="1"
                        max="50"
                        step="1"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Averages reduce noise but increase total acquisition time.
                      </p>
                    </div>

                    {/* Dark Subtraction Controls */}
                    <div className="border border-gray-200 rounded-md p-3 bg-gray-50">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-700">Dark Subtraction</span>
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={useDarkSubtraction}
                            onChange={(e) => setUseDarkSubtraction(e.target.checked)}
                            className="rounded"
                          />
                          <span className="text-xs text-gray-600">Enable</span>
                        </label>
                      </div>
                      {hasDark ? (
                        <div className="text-xs text-green-700 bg-green-50 p-2 rounded mb-2">
                          Dark captured: {darkInfo?.data_points} pts, mean={darkInfo?.mean?.toFixed(0)}
                        </div>
                      ) : (
                        <div className="text-xs text-yellow-700 bg-yellow-50 p-2 rounded mb-2">
                          No dark spectrum. Acquire one for better results.
                        </div>
                      )}
                      <div className="flex gap-2">
                        <button
                          onClick={handleAcquireDark}
                          disabled={acquiringDark}
                          className="flex-1 px-2 py-1 text-xs bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50"
                        >
                          {acquiringDark ? 'Acquiring...' : 'Acquire Dark'}
                        </button>
                        {hasDark && (
                          <button
                            onClick={handleClearDark}
                            className="px-2 py-1 text-xs text-red-600 border border-red-300 rounded hover:bg-red-50"
                          >
                            Clear
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Calibration Controls */}
                    <div className="border border-gray-200 rounded-md p-3 bg-gray-50 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-gray-700">Calibration</span>
                        <button
                          onClick={() => {
                            void loadCalibrations(instrumentId)
                            void loadActiveCalibration(instrumentId)
                          }}
                          className="text-xs text-primary-600 hover:text-primary-700"
                        >
                          Refresh
                        </button>
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-1">
                          Instrument ID
                        </label>
                        <input
                          type="text"
                          value={instrumentId}
                          onChange={(e) => setInstrumentId(e.target.value || 'default')}
                          className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary-500"
                        />
                        <p className="text-[11px] text-gray-500 mt-1">
                          Use the serial port or a stable label to separate instruments.
                        </p>
                      </div>

                      <div className="text-xs text-gray-700">
                        Active calibration:{' '}
                        <span className="font-medium">
                          {activeCalibration?.name ?? 'None'}
                        </span>
                      </div>

                      {calibrationList.length > 0 ? (
                        <div className="flex items-center gap-2">
                          <select
                            value={selectedCalibrationId}
                            onChange={(e) =>
                              setSelectedCalibrationId(
                                e.target.value ? Number(e.target.value) : ''
                              )
                            }
                            className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded"
                          >
                            <option value="">Select calibration…</option>
                            {calibrationList.map((calibration) => (
                              <option key={calibration.id} value={calibration.id}>
                                {calibration.name}
                                {calibration.active ? ' (active)' : ''}
                              </option>
                            ))}
                          </select>
                          <button
                            onClick={() => {
                              if (selectedCalibrationId) {
                                void handleActivateCalibration(Number(selectedCalibrationId))
                              }
                            }}
                            className="px-2 py-1 text-xs border border-gray-300 rounded hover:bg-gray-100"
                          >
                            Activate
                          </button>
                        </div>
                      ) : (
                        <div className="text-xs text-gray-500">
                          No saved calibrations yet.
                        </div>
                      )}

                      <div className="border-t border-gray-200 pt-3 space-y-2">
                        <div className="text-xs font-medium text-gray-700">
                          Create calibration (two-point)
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <input
                            type="text"
                            value={calibrationName}
                            onChange={(e) => setCalibrationName(e.target.value)}
                            placeholder="Calibration name"
                            className="col-span-2 px-2 py-1 text-xs border border-gray-300 rounded"
                          />
                          <input
                            type="number"
                            value={pixelA}
                            onChange={(e) => setPixelA(e.target.value)}
                            placeholder="Pixel A (0-based)"
                            className="px-2 py-1 text-xs border border-gray-300 rounded"
                          />
                          <input
                            type="number"
                            value={wavenumberA}
                            onChange={(e) => setWavenumberA(e.target.value)}
                            placeholder="Shift A (cm⁻¹)"
                            className="px-2 py-1 text-xs border border-gray-300 rounded"
                          />
                          <input
                            type="number"
                            value={pixelB}
                            onChange={(e) => setPixelB(e.target.value)}
                            placeholder="Pixel B (0-based)"
                            className="px-2 py-1 text-xs border border-gray-300 rounded"
                          />
                          <input
                            type="number"
                            value={wavenumberB}
                            onChange={(e) => setWavenumberB(e.target.value)}
                            placeholder="Shift B (cm⁻¹)"
                            className="px-2 py-1 text-xs border border-gray-300 rounded"
                          />
                          <input
                            type="number"
                            value={calibrationLaser}
                            onChange={(e) => setCalibrationLaser(e.target.value)}
                            placeholder="Laser nm (optional)"
                            className="px-2 py-1 text-xs border border-gray-300 rounded"
                          />
                          <input
                            type="text"
                            value={calibrationNotes}
                            onChange={(e) => setCalibrationNotes(e.target.value)}
                            placeholder="Notes (optional)"
                            className="px-2 py-1 text-xs border border-gray-300 rounded col-span-2"
                          />
                        </div>
                        {calibrationError && (
                          <div className="text-[11px] text-red-600">{calibrationError}</div>
                        )}
                        <button
                          onClick={handleCreateCalibration}
                          disabled={calibrationBusy}
                          className="w-full px-2 py-1 text-xs bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50"
                        >
                          {calibrationBusy ? 'Saving...' : 'Save Calibration'}
                        </button>
                      </div>
                    </div>

                    <button
                      onClick={handleAcquireSpectrum}
                      disabled={acquiring || loading}
                      className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {acquiring ? '🔬 Acquiring Spectrum...' : hasDark && useDarkSubtraction ? '🔬 Acquire & Analyze (Dark Corrected)' : '🔬 Acquire & Analyze'}
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
                {acquisitionInfo && (
                  <div
                    className={`border rounded-md p-3 text-sm ${
                      acquisitionInfo.source === 'hardware' || acquisitionInfo.source === 'hardware_corrected'
                        ? 'border-green-200 bg-green-50'
                        : 'border-blue-200 bg-blue-50'
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="font-medium">
                        {acquisitionInfo.source === 'hardware'
                          ? 'Hardware spectrum'
                          : acquisitionInfo.source === 'hardware_corrected'
                            ? 'Dark-corrected spectrum'
                            : acquisitionInfo.source === 'dark'
                              ? 'Dark spectrum'
                              : 'Simulator spectrum'}
                      </span>
                      <span className="text-xs text-gray-600">
                        Captured {formatTimestamp(acquisitionInfo.acquired_at)}
                      </span>
                    </div>
                    <div className="mt-2 text-xs text-gray-700 space-y-1">
                      <div>Integration: {acquisitionInfo.integration_time} ms</div>
                      {acquisitionInfo.average_count && acquisitionInfo.average_count > 1 && (
                        <div>Averages: {acquisitionInfo.average_count}</div>
                      )}
                      {(acquisitionInfo.source === 'hardware' ||
                        acquisitionInfo.source === 'hardware_corrected') &&
                        acquisitionInfo.port && (
                        <div>Port: {acquisitionInfo.port}</div>
                      )}
                      {acquisitionInfo.source === 'simulator' && acquisitionInfo.simulation_file && (
                        <div>Source file: {acquisitionInfo.simulation_file}</div>
                      )}
                    </div>
                  </div>
                )}

                {Array.isArray(rawSpectrumData) && rawSpectrumData.length > 0 && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Acquired Spectrum (first 10 points)
                    </label>
                    <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                      {rawSpectrumData.slice(0, 10).map((value) => value.toFixed(2)).join(', ')}...
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

                {/* Spectral Charts */}
                {rawSpectrumData.length > 0 && (
                  <div className="bg-white border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-4">Raw Spectrum</h4>
                    <SpectralChart
                      spectrumData={rawSpectrumData}
                      xAxisData={activeCalibration?.axis_data}
                      compoundName={acquisitionInfo?.source === 'simulator' ? 'Simulator Capture' : 'Hardware Capture'}
                      showPeaks={false}
                      height={320}
                      color="rgb(59, 130, 246)"
                      backgroundColor="rgba(59, 130, 246, 0.08)"
                    />
                  </div>
                )}

                {processedSpectrumData.length > 0 && (
                  <div className="bg-white border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-4">
                      Processed Spectrum
                      {acquisitionInfo && (
                        <span className="ml-2 text-xs uppercase tracking-wide text-gray-500">
                          {acquisitionInfo.source}
                        </span>
                      )}
                    </h4>
                    <SpectralChart
                      spectrumData={processedSpectrumData}
                      xAxisData={activeCalibration?.axis_data}
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
                      <span>
                        {result.individual_predictions?.random_forest?.compound ?? 'N/A'} ({((result.individual_predictions?.random_forest?.confidence ?? 0) * 100).toFixed(1)}%)
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>SVM:</span>
                      <span>
                        {result.individual_predictions?.svm?.compound ?? 'N/A'} ({((result.individual_predictions?.svm?.confidence ?? 0) * 100).toFixed(1)}%)
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Neural Network:</span>
                      <span>
                        {result.individual_predictions?.neural_network?.compound ?? 'N/A'} ({((result.individual_predictions?.neural_network?.confidence ?? 0) * 100).toFixed(1)}%)
                      </span>
                    </div>
                    {result.individual_predictions?.pls_regression && (
                      <div className="flex justify-between">
                        <span>PLS Regression:</span>
                        <span>
                          {result.individual_predictions.pls_regression.compound} ({((result.individual_predictions.pls_regression.confidence ?? 0) * 100).toFixed(1)}%)
                        </span>
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

                {referenceSpectra.length > 0 && (
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Reference Spectra (beta)</h4>
                    <div className="grid grid-cols-1 gap-4">
                      {referenceSpectra.slice(0, 2).map((reference) => {
                        const previewData =
                          reference.preprocessed_spectrum && reference.preprocessed_spectrum.length > 0
                            ? reference.preprocessed_spectrum
                            : reference.spectrum_data
                        const subtitle =
                          reference.measurement_conditions ||
                          (reference.metadata?.source as string | undefined) ||
                          'Reference library match'

                        return (
                          <SpectrumPreview
                            key={`reference-${reference.id}`}
                            title={reference.compound_name}
                            subtitle={subtitle}
                            spectrumData={previewData}
                            showPeaks={false}
                            color="rgb(99, 102, 241)"
                            backgroundColor="rgba(99, 102, 241, 0.1)"
                            footer={
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                <span>
                                  <span className="font-semibold">Laser:</span>{' '}
                                  {reference.laser_wavelength ?? '—'} nm
                                </span>
                                <span>
                                  <span className="font-semibold">Integration:</span>{' '}
                                  {reference.integration_time ?? '—'} ms
                                </span>
                                {reference.metadata?.dataset_zip && (
                                  <span className="col-span-2 truncate">
                                    <span className="font-semibold">Source:</span>{' '}
                                    {reference.metadata.dataset_zip}
                                  </span>
                                )}
                              </div>
                            }
                          />
                        )
                      })}
                    </div>
                    {referenceSpectra.length > 2 && (
                      <p className="mt-2 text-xs text-gray-500">
                        Showing first 2 matches. Refine hints to narrow results.
                      </p>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>Enter spectrum data and click &#39;Analyze&#39; to see results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
