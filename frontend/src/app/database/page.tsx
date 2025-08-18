'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/utils/api'
import { DatabaseStats, Spectrum } from '@/types/spectrum'

export default function DatabasePage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Spectrum[]>([])
  const [searching, setSearching] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [browsing, setBrowsing] = useState(false)
  const [selectedSpectrum, setSelectedSpectrum] = useState<Spectrum | null>(null)
  const [showSpectrumModal, setShowSpectrumModal] = useState(false)

  const { data: stats, isLoading: statsLoading } = useQuery<DatabaseStats>({
    queryKey: ['database-stats'],
    queryFn: () => api.getDatabaseStats()
  })

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    
    setSearching(true)
    try {
      const results = await api.searchSpectra(searchQuery)
      setSearchResults(results)
      setShowResults(true)
    } catch (error) {
      console.error('Search failed:', error)
      alert('Search failed. Please try again.')
    } finally {
      setSearching(false)
    }
  }

  const handleBrowseAll = async () => {
    setBrowsing(true)
    try {
      // Search with empty query to get all results
      const results = await api.searchSpectra('')
      setSearchResults(results)
      setShowResults(true)
    } catch (error) {
      console.error('Browse all failed:', error)
      alert('Failed to browse database. Please try again.')
    } finally {
      setBrowsing(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  const handleViewSpectrum = (spectrum: Spectrum) => {
    setSelectedSpectrum(spectrum)
    setShowSpectrumModal(true)
  }

  const handleAnalyzeSpectrum = async (spectrum: Spectrum) => {
    if (!spectrum.spectrum_data || spectrum.spectrum_data.length === 0) {
      alert('No spectrum data available for analysis')
      return
    }

    try {
      // Navigate to analyze page with spectrum data
      // For now, we'll copy the data to clipboard and show instructions
      const spectrumDataStr = spectrum.spectrum_data.join(', ')
      await navigator.clipboard.writeText(spectrumDataStr)
      alert(`Spectrum data copied to clipboard! Go to the Analyze page and paste it in manual mode.\n\nCompound: ${spectrum.compound_name}`)
    } catch (error) {
      console.error('Failed to copy spectrum data:', error)
      alert('Failed to copy spectrum data. Please try again.')
    }
  }

  const handleQuickFilter = async (filterType: string) => {
    let query = ''
    switch (filterType) {
      case 'minerals':
        query = 'RRUFF' // Search for RRUFF data which are mostly minerals
        break
      case 'organics':
        query = 'synthetic' // Search for synthetic organic compounds
        break
      case 'recent':
        // Get all results and they'll be in recent order
        query = ''
        break
      default:
        return
    }

    try {
      setSearching(true)
      const results = await api.searchSpectra(query)
      setSearchResults(results)
      setShowResults(true)
      setSearchQuery(query)
    } catch (error) {
      console.error('Filter failed:', error)
      alert('Filter failed. Please try again.')
    } finally {
      setSearching(false)
    }
  }

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
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
              
              <div className="grid grid-cols-2 gap-4">
                <button 
                  onClick={handleSearch}
                  disabled={searching || !searchQuery.trim()}
                  className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {searching ? 'Searching...' : 'Search'}
                </button>
                <button 
                  onClick={handleBrowseAll}
                  disabled={browsing}
                  className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {browsing ? 'Loading...' : 'Browse All'}
                </button>
              </div>
            </div>
            
            <div className="mt-6">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Quick Filters</h3>
              <div className="flex flex-wrap gap-2">
                <button 
                  onClick={() => handleQuickFilter('minerals')}
                  disabled={searching}
                  className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200 disabled:opacity-50"
                >
                  Minerals
                </button>
                <button 
                  onClick={() => handleQuickFilter('organics')}
                  disabled={searching}
                  className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200 disabled:opacity-50"
                >
                  Organics
                </button>
                <button 
                  onClick={() => handleQuickFilter('recent')}
                  disabled={searching}
                  className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200 disabled:opacity-50"
                >
                  Recent
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Search Results */}
      {showResults && (
        <div className="mt-8">
          <div className="card">
            <div className="card-header flex justify-between items-center">
              <h2 className="text-lg font-medium">
                Search Results ({searchResults.length} found)
              </h2>
              <button
                onClick={() => setShowResults(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="card-content">
              {searchResults.length > 0 ? (
                <div className="space-y-4">
                  {searchResults.map((spectrum) => (
                    <div key={spectrum.id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="text-lg font-medium text-gray-900">
                            {spectrum.compound_name}
                          </h3>
                          {spectrum.chemical_formula && (
                            <p className="text-sm text-gray-600 mt-1">
                              Formula: {spectrum.chemical_formula}
                            </p>
                          )}
                          {spectrum.cas_number && (
                            <p className="text-sm text-gray-600">
                              CAS: {spectrum.cas_number}
                            </p>
                          )}
                          <div className="flex items-center space-x-4 mt-2 text-sm text-gray-500">
                            <span>
                              {spectrum.spectrum_data?.length || 0} data points
                            </span>
                            <span>
                              {spectrum.laser_wavelength}nm laser
                            </span>
                            <span>
                              {spectrum.source || 'Unknown source'}
                            </span>
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <button 
                            onClick={() => handleViewSpectrum(spectrum)}
                            className="px-3 py-1 text-sm bg-primary-100 text-primary-700 rounded hover:bg-primary-200"
                          >
                            View
                          </button>
                          <button 
                            onClick={() => handleAnalyzeSpectrum(spectrum)}
                            className="px-3 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
                          >
                            Analyze
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p>No spectra found matching your search.</p>
                  <p className="text-sm mt-1">Try a different search term or browse all.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

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

      {/* Spectrum View Modal */}
      {showSpectrumModal && selectedSpectrum && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-lg font-bold text-gray-900">
                  {selectedSpectrum.compound_name}
                </h3>
                {selectedSpectrum.chemical_formula && (
                  <p className="text-sm text-gray-600 mt-1">
                    Formula: {selectedSpectrum.chemical_formula}
                  </p>
                )}
              </div>
              <button
                onClick={() => setShowSpectrumModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-700">ID:</span>
                  <span className="ml-2">{selectedSpectrum.id}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Source:</span>
                  <span className="ml-2">{selectedSpectrum.source}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Data Points:</span>
                  <span className="ml-2">{selectedSpectrum.spectrum_data?.length || 0}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Laser Wavelength:</span>
                  <span className="ml-2">{selectedSpectrum.laser_wavelength || 'N/A'}nm</span>
                </div>
                {selectedSpectrum.cas_number && (
                  <div className="col-span-2">
                    <span className="font-medium text-gray-700">CAS Number:</span>
                    <span className="ml-2">{selectedSpectrum.cas_number}</span>
                  </div>
                )}
              </div>

              {selectedSpectrum.measurement_conditions && (
                <div>
                  <span className="font-medium text-gray-700">Conditions:</span>
                  <p className="text-sm text-gray-600 mt-1">{selectedSpectrum.measurement_conditions}</p>
                </div>
              )}

              {selectedSpectrum.spectrum_data && selectedSpectrum.spectrum_data.length > 0 && (
                <div>
                  <span className="font-medium text-gray-700">Spectrum Data Preview:</span>
                  <div className="mt-2 p-2 bg-gray-100 rounded text-xs font-mono max-h-32 overflow-y-auto">
                    {selectedSpectrum.spectrum_data.slice(0, 50).join(', ')}
                    {selectedSpectrum.spectrum_data.length > 50 && '...'}
                  </div>
                </div>
              )}

              <div className="flex justify-end space-x-3 mt-6">
                <button
                  onClick={() => setShowSpectrumModal(false)}
                  className="px-4 py-2 text-gray-700 border border-gray-300 rounded hover:bg-gray-50"
                >
                  Close
                </button>
                <button
                  onClick={() => {
                    handleAnalyzeSpectrum(selectedSpectrum)
                    setShowSpectrumModal(false)
                  }}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Analyze This Spectrum
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}