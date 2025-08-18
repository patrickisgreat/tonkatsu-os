'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/utils/api'

export default function TrainingPage() {
  const [isTraining, setIsTraining] = useState(false)
  const [trainingConfig, setTrainingConfig] = useState({
    use_pca: true,
    n_components: 50,
    validation_split: 0.2,
    optimize_hyperparams: false
  })

  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ['training-status'],
    queryFn: () => api.getTrainingStatus(),
    refetchInterval: isTraining ? 2000 : false
  })

  const startTraining = async () => {
    setIsTraining(true)
    try {
      const result = await api.trainModel(trainingConfig)
      console.log('Training result:', result)
      await refetchStatus()
    } catch (error) {
      console.error('Training failed:', error)
      alert('Training failed. Please check the console for details.')
    } finally {
      setIsTraining(false)
    }
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

                {status.progress !== null && (
                  <div>
                    <div className="flex justify-between text-sm text-gray-700 mb-1">
                      <span>Progress</span>
                      <span>{(status.progress * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${status.progress * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {status.last_trained && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">Last Trained:</span>
                    <span className="text-sm text-gray-600">
                      {new Date(status.last_trained).toLocaleString()}
                    </span>
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
    </div>
  )
}