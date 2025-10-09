'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { SpectrumPreview } from '@/components/SpectrumPreview'
import { api } from '@/utils/api'
import type { DatabaseStats, Spectrum, TrainingResult, TrainingStatus } from '@/types/spectrum'

export default function TrainingPage() {
  const [isTraining, setIsTraining] = useState(false)
  const [trainingConfig, setTrainingConfig] = useState({
    use_pca: true,
    n_components: 50,
    validation_split: 0.2,
    optimize_hyperparams: false
  })
  const [lastTrainingResult, setLastTrainingResult] = useState<TrainingResult | null>(null)

  const formatTimestamp = (value?: string) => {
    if (!value) return '—'
    const date = new Date(value)
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
  }

  const formatPercent = (value?: number) =>
    typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'

  const formatSeconds = (value?: number) =>
    typeof value === 'number' ? `${value.toFixed(1)} s` : '—'

  const {
    data: status,
    refetch: refetchStatus,
  } = useQuery<TrainingStatus>({
    queryKey: ['training-status'],
    queryFn: () => api.getTrainingStatus(),
    refetchInterval: isTraining ? 2000 : false
  })

  const {
    data: modelMetrics,
    isLoading: metricsLoading,
    error: metricsError,
    refetch: refetchMetrics,
  } = useQuery<Record<string, any>>({
    queryKey: ['model-metrics'],
    queryFn: () => api.getModelMetrics(),
  })

  const {
    data: stats,
    isLoading: statsLoading,
    error: statsError,
  } = useQuery<DatabaseStats>({
    queryKey: ['database-stats'],
    queryFn: () => api.getDatabaseStats(),
  })

  const topCompound = stats?.top_compounds?.[0]?.[0] as string | undefined

  const {
    data: previewSpectrum,
    isLoading: previewLoading,
    error: previewError,
  } = useQuery<Spectrum | null>({
    queryKey: ['preview-spectrum', topCompound],
    queryFn: async () => {
      if (!topCompound) return null
      const results = await api.searchSpectra(topCompound)
      return results[0] ?? null
    },
    enabled: Boolean(topCompound),
  })

  const previewSpectrumData =
    previewSpectrum?.preprocessed_spectrum && previewSpectrum.preprocessed_spectrum.length > 0
      ? previewSpectrum.preprocessed_spectrum
      : previewSpectrum?.spectrum_data

  const topCompoundCount = stats?.top_compounds?.[0]?.[1] as number | undefined
  const datasetEmpty = (stats?.total_spectra ?? 0) === 0
  const ensembleAccuracy =
    typeof lastTrainingResult?.ensemble_accuracy === 'number'
      ? lastTrainingResult.ensemble_accuracy
      : typeof modelMetrics?.ensemble_accuracy === 'number'
      ? (modelMetrics.ensemble_accuracy as number)
      : undefined
  const rfAccuracy =
    typeof lastTrainingResult?.rf_accuracy === 'number'
      ? lastTrainingResult.rf_accuracy
      : typeof modelMetrics?.rf_accuracy === 'number'
      ? (modelMetrics.rf_accuracy as number)
      : undefined
  const svmAccuracy =
    typeof lastTrainingResult?.svm_accuracy === 'number'
      ? lastTrainingResult.svm_accuracy
      : typeof modelMetrics?.svm_accuracy === 'number'
      ? (modelMetrics.svm_accuracy as number)
      : undefined
  const nnAccuracy =
    typeof lastTrainingResult?.nn_accuracy === 'number'
      ? lastTrainingResult.nn_accuracy
      : typeof modelMetrics?.nn_accuracy === 'number'
      ? (modelMetrics.nn_accuracy as number)
      : undefined
  const plsAccuracy =
    typeof lastTrainingResult?.pls_accuracy === 'number'
      ? lastTrainingResult.pls_accuracy
      : typeof modelMetrics?.pls_accuracy === 'number'
      ? (modelMetrics.pls_accuracy as number)
      : undefined
  const trainingTime =
    typeof lastTrainingResult?.training_time === 'number'
      ? lastTrainingResult.training_time
      : typeof modelMetrics?.training_time === 'number'
      ? (modelMetrics.training_time as number)
      : undefined
  const nTrainSamples =
    lastTrainingResult?.n_train_samples ?? (modelMetrics?.n_train_samples as number | undefined)
  const nValSamples =
    lastTrainingResult?.n_val_samples ?? (modelMetrics?.n_val_samples as number | undefined)
  const nClasses =
    lastTrainingResult?.n_classes ?? (modelMetrics?.n_classes as number | undefined)
  const metricsTimestamp =
    (typeof modelMetrics?.timestamp === 'string' ? modelMetrics.timestamp : undefined) ??
    status?.last_trained

  const startTraining = async () => {
    setIsTraining(true)
    try {
      const result = await api.trainModels(trainingConfig)
      setLastTrainingResult(result)
      await Promise.all([refetchStatus(), refetchMetrics()])
    } catch (error) {
      console.error('Training failed:', error)
      alert('Training failed. Please check the console for details.')
    } finally {
      setIsTraining(false)
    }
  }

  const renderPerformanceContent = () => {
    if (metricsLoading) {
      return (
        <div className="flex items-center justify-center py-10 text-sm text-gray-500">
          Loading performance metrics...
        </div>
      )
    }

    if (metricsError) {
      return (
        <div className="text-sm text-red-600">
          Failed to load metrics. Please try again later.
        </div>
      )
    }

    if (!modelMetrics && !lastTrainingResult) {
      return (
        <div className="text-sm text-gray-500">
          Train the model to generate performance metrics.
        </div>
      )
    }

    return (
      <>
        <div className="grid grid-cols-2 gap-4 text-center">
          <div className="rounded-lg bg-primary-50 p-4">
            <p className="text-sm text-gray-600">Ensemble</p>
            <p className="text-2xl font-semibold text-primary-700">
              {formatPercent(ensembleAccuracy)}
            </p>
          </div>
          <div className="rounded-lg bg-blue-50 p-4">
            <p className="text-sm text-gray-600">Random Forest</p>
            <p className="text-2xl font-semibold text-blue-700">
              {formatPercent(rfAccuracy)}
            </p>
          </div>
          <div className="rounded-lg bg-green-50 p-4">
            <p className="text-sm text-gray-600">SVM</p>
            <p className="text-2xl font-semibold text-green-700">
              {formatPercent(svmAccuracy)}
            </p>
          </div>
          <div className="rounded-lg bg-purple-50 p-4">
            <p className="text-sm text-gray-600">Neural Network</p>
            <p className="text-2xl font-semibold text-purple-700">
              {formatPercent(nnAccuracy)}
            </p>
          </div>
          {plsAccuracy !== undefined && (
            <div className="rounded-lg bg-yellow-50 p-4 col-span-2">
              <p className="text-sm text-gray-600">PLS Regression</p>
              <p className="text-2xl font-semibold text-yellow-700">
                {formatPercent(plsAccuracy)}
              </p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm text-gray-700">
          <div className="rounded-lg bg-gray-50 p-3">
            <p className="font-medium text-gray-900">Training Time</p>
            <p>{formatSeconds(trainingTime)}</p>
          </div>
          <div className="rounded-lg bg-gray-50 p-3">
            <p className="font-medium text-gray-900">Classes</p>
            <p>{nClasses ?? '—'}</p>
          </div>
          <div className="rounded-lg bg-gray-50 p-3">
            <p className="font-medium text-gray-900">Train Samples</p>
            <p>{nTrainSamples ?? '—'}</p>
          </div>
          <div className="rounded-lg bg-gray-50 p-3">
            <p className="font-medium text-gray-900">Validation Samples</p>
            <p>{nValSamples ?? '—'}</p>
          </div>
        </div>

        {lastTrainingResult && (
          <p className="text-xs text-gray-500">
            Metrics reflect the most recent training run.
          </p>
        )}
      </>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Model Training</h1>
        <p className="text-gray-600 mt-2">Train and optimize machine learning models for spectral analysis</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Training Configuration */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Training Configuration</h2>
          </div>
          <div className="card-content space-y-4">
            <div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={trainingConfig.use_pca}
                  onChange={(e) => setTrainingConfig({
                    ...trainingConfig,
                    use_pca: e.target.checked
                  })}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="ml-2 text-sm text-gray-700">Use PCA for dimensionality reduction</span>
              </label>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of PCA Components
              </label>
              <input
                type="number"
                min="10"
                max="200"
                value={trainingConfig.n_components}
                onChange={(e) => setTrainingConfig({
                  ...trainingConfig,
                  n_components: parseInt(e.target.value)
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Validation Split
              </label>
              <input
                type="number"
                min="0.1"
                max="0.4"
                step="0.05"
                value={trainingConfig.validation_split}
                onChange={(e) => setTrainingConfig({
                  ...trainingConfig,
                  validation_split: parseFloat(e.target.value)
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>

            <div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={trainingConfig.optimize_hyperparams}
                  onChange={(e) => setTrainingConfig({
                    ...trainingConfig,
                    optimize_hyperparams: e.target.checked
                  })}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="ml-2 text-sm text-gray-700">Optimize hyperparameters (slower)</span>
              </label>
            </div>

            <button
              onClick={startTraining}
              disabled={isTraining || status?.is_training}
              className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isTraining || status?.is_training ? 'Training in Progress...' : 'Start Training'}
            </button>
          </div>
        </div>

        {/* Training Status */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Training Status</h2>
          </div>
          <div className="card-content">
            {status ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Status:</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    status.is_training ? 'bg-blue-100 text-blue-800' : 
                    status.model_exists ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {status.is_training ? 'Training' : status.model_exists ? 'Model Ready' : 'No Model'}
                  </span>
                </div>

                {status.progress != null && (
                  <div>
                    <div className="flex justify-between text-sm text-gray-700 mb-1">
                      <span>Progress</span>
                      <span>{((status.progress || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${(status.progress || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {status.last_trained && (
                  <div className="text-xs text-gray-500">
                    Last trained: {formatTimestamp(status.last_trained)}
                  </div>
                )}

              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>Loading training status...</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Information */}
      <div className="mt-8">
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Model Architecture</h2>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-2">
                  <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </div>
                <h3 className="font-medium text-gray-900">Random Forest</h3>
                <p className="text-sm text-gray-600">Ensemble of decision trees</p>
              </div>

              <div className="text-center">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-2">
                  <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                  </svg>
                </div>
                <h3 className="font-medium text-gray-900">SVM</h3>
                <p className="text-sm text-gray-600">Support Vector Machine</p>
              </div>

              <div className="text-center">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-2">
                  <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="font-medium text-gray-900">Neural Network</h3>
                <p className="text-sm text-gray-600">Multi-layer perceptron</p>
              </div>

              <div className="text-center">
                <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center mx-auto mb-2">
                  <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="font-medium text-gray-900">PLS Regression</h3>
                <p className="text-sm text-gray-600">Partial Least Squares</p>
              </div>
            </div>
          </div>
        </div>
      </div>

    <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div className="card">
        <div className="card-header">
          <div>
            <h2 className="text-lg font-medium">Model Performance</h2>
            {metricsTimestamp && (
              <p className="text-sm text-gray-600 mt-1">
                Last update: {formatTimestamp(metricsTimestamp)}
              </p>
            )}
          </div>
        </div>
        <div className="card-content space-y-4">{renderPerformanceContent()}</div>
      </div>

      <SpectrumPreview
        className="h-full"
        title={topCompound ? `Training Preview: ${topCompound}` : 'Training Dataset Preview'}
        subtitle={
          topCompound
            ? `Most represented compound${topCompoundCount ? ` (${topCompoundCount} spectra)` : ''}`
            : datasetEmpty
            ? 'Import spectra to start training'
            : 'Add spectra or select a compound to view a preview'
        }
        spectrumData={previewSpectrumData}
        loading={statsLoading || previewLoading}
        emptyMessage={
          previewError
            ? 'Unable to load preview spectrum.'
            : datasetEmpty
            ? 'Add spectra to your database to preview training data.'
            : 'Preview unavailable for this compound.'
        }
        footer={
          previewSpectrum ? (
            <div className="grid grid-cols-2 gap-2 text-xs">
              <span>
                <span className="font-semibold">Laser:</span> {previewSpectrum.laser_wavelength ?? '—'} nm
              </span>
              <span>
                <span className="font-semibold">Integration:</span> {previewSpectrum.integration_time ?? '—'} ms
              </span>
              <span className="col-span-2">
                <span className="font-semibold">Source:</span> {previewSpectrum.source}
              </span>
            </div>
          ) : undefined
        }
      />
    </div>
  </div>
  )
}
