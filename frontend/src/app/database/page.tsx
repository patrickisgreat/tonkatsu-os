'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/utils/api'
import { DatabaseStats } from '@/types/spectrum'

export default function DatabasePage() {
  const { data: stats, isLoading: statsLoading } = useQuery<DatabaseStats>({
    queryKey: ['database-stats'],
    queryFn: () => api.getDatabaseStats()
  })

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Database Explorer</h1>
        <p className="text-gray-600 mt-2">Browse and manage your spectral database</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Spectra</p>
                <p className="text-2xl font-bold text-gray-900">
                  {statsLoading ? '...' : stats?.total_spectra || 0}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-green-500 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Unique Compounds</p>
                <p className="text-2xl font-bold text-gray-900">
                  {statsLoading ? '...' : stats?.unique_compounds || 0}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-purple-500 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Database Size</p>
                <p className="text-2xl font-bold text-gray-900">
                  {stats?.total_spectra ? `${(stats.total_spectra * 0.1).toFixed(1)} MB` : '0 MB'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top Compounds */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Top Compounds</h2>
          </div>
          <div className="card-content">
            {stats?.top_compounds?.length ? (
              <div className="space-y-3">
                {stats.top_compounds.map(([compound, count], index) => (
                  <div key={compound} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="w-6 h-6 bg-primary-100 text-primary-700 rounded-full flex items-center justify-center text-sm font-medium mr-3">
                        {index + 1}
                      </span>
                      <span className="text-gray-900">{compound}</span>
                    </div>
                    <span className="text-gray-500">{count} spectra</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>No compounds in database</p>
                <p className="text-sm mt-1">Import some spectra to get started</p>
              </div>
            )}
          </div>
        </div>

        {/* Search */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Search Database</h2>
          </div>
          <div className="card-content">
            <div className="space-y-4">
              <input
                type="text"
                placeholder="Search compounds..."
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
              
              <div className="grid grid-cols-2 gap-4">
                <button className="btn-primary">
                  Search
                </button>
                <button className="btn-secondary">
                  Browse All
                </button>
              </div>
            </div>
            
            <div className="mt-6">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Quick Filters</h3>
              <div className="flex flex-wrap gap-2">
                <button className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200">
                  Minerals
                </button>
                <button className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200">
                  Organics
                </button>
                <button className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200">
                  Recent
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Database Actions */}
      <div className="mt-8">
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium">Database Management</h2>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <button className="btn-secondary">
                Export Database
              </button>
              <button className="btn-secondary">
                Backup Database
              </button>
              <button className="btn-secondary text-red-600 border-red-300 hover:bg-red-50">
                Reset Database
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}