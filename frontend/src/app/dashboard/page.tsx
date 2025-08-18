'use client'

import { useQuery } from '@tanstack/react-query'
import { CircleStackIcon as DatabaseIcon, BeakerIcon, CpuChipIcon, ChartBarIcon } from '@heroicons/react/24/outline'
import { api } from '@/utils/api'
import { DatabaseStats } from '@/types/spectrum'

export default function DashboardPage() {
  const { data: stats, isLoading, error } = useQuery<DatabaseStats>({
    queryKey: ['database-stats'],
    queryFn: () => api.getDatabaseStats(),
  })

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600 mb-4">Connection Error</h1>
          <p className="text-gray-600 mb-4">
            Unable to connect to the backend. Make sure the API server is running on port 8000.
          </p>
          <button 
            onClick={() => window.location.reload()} 
            className="btn-primary"
          >
            Retry Connection
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Welcome to your Raman analysis control center
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="stat-card">
            <div className="stat-card-content">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <DatabaseIcon className="h-8 w-8 text-blue-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="stat-label">Total Spectra</dt>
                    <dd className="stat-value">{stats?.total_spectra || 0}</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-card-content">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <BeakerIcon className="h-8 w-8 text-green-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="stat-label">Unique Compounds</dt>
                    <dd className="stat-value">{stats?.unique_compounds || 0}</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-card-content">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <CpuChipIcon className="h-8 w-8 text-purple-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="stat-label">ML Models</dt>
                    <dd className="stat-value">3</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-card-content">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <ChartBarIcon className="h-8 w-8 text-orange-600" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="stat-label">Analyses Today</dt>
                    <dd className="stat-value">0</dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Quick Actions</h3>
            </div>
            <div className="card-content">
              <div className="space-y-4">
                <a
                  href="/analyze"
                  className="block w-full btn-primary text-center"
                >
                  🔬 Analyze New Spectrum
                </a>
                <a
                  href="/import"
                  className="block w-full btn-secondary text-center"
                >
                  📥 Import Spectral Data
                </a>
                <a
                  href="/training"
                  className="block w-full btn-secondary text-center"
                >
                  🤖 Train ML Models
                </a>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Recent Activity</h3>
            </div>
            <div className="card-content">
              <div className="text-center py-8 text-gray-500">
                <ChartBarIcon className="mx-auto h-12 w-12 mb-4 opacity-50" />
                <p>No recent activity</p>
                <p className="text-sm">Start analyzing spectra to see activity here</p>
              </div>
            </div>
          </div>
        </div>

        {/* Top Compounds */}
        {stats && stats.top_compounds && stats.top_compounds.length > 0 && (
          <div className="mt-8">
            <div className="card">
              <div className="card-header">
                <h3 className="text-lg font-medium text-gray-900">Top Compounds in Database</h3>
              </div>
              <div className="card-content">
                <div className="space-y-2">
                  {stats.top_compounds.slice(0, 10).map(([compound, count], index) => (
                    <div key={compound} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-b-0">
                      <span className="font-medium text-gray-900">{compound}</span>
                      <span className="badge badge-info">{count} spectra</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}