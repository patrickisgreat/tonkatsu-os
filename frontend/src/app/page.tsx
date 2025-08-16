'use client'

import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { 
  BeakerIcon, 
  ChartBarIcon, 
  CpuChipIcon, 
  DocumentArrowUpIcon,
  EyeIcon,
  DatabaseIcon
} from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'
import { api } from '@/utils/api'
import { DatabaseStats } from '@/types/spectrum'

const features = [
  {
    name: 'Live Analysis',
    description: 'Real-time spectrum acquisition and AI-powered molecular identification',
    icon: EyeIcon,
    href: '/analyze',
    color: 'bg-blue-500',
  },
  {
    name: 'Import Spectra',
    description: 'Upload and process spectral data from various file formats',
    icon: DocumentArrowUpIcon,
    href: '/import',
    color: 'bg-green-500',
  },
  {
    name: 'Database Explorer',
    description: 'Browse, search, and manage your spectral database',
    icon: DatabaseIcon,
    href: '/database',
    color: 'bg-purple-500',
  },
  {
    name: 'Visualizations',
    description: 'Interactive plots and comprehensive data analysis',
    icon: ChartBarIcon,
    href: '/visualizations',
    color: 'bg-yellow-500',
  },
  {
    name: 'Model Training',
    description: 'Train and optimize machine learning models',
    icon: CpuChipIcon,
    href: '/training',
    color: 'bg-red-500',
  },
  {
    name: 'Research Tools',
    description: 'Advanced analysis tools for research applications',
    icon: BeakerIcon,
    href: '/research',
    color: 'bg-indigo-500',
  },
]

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
}

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 300,
      damping: 24,
    },
  },
}

export default function HomePage() {
  const { data: stats, isLoading: statsLoading } = useQuery<DatabaseStats>({
    queryKey: ['database-stats'],
    queryFn: () => api.getDatabaseStats(),
  })

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-primary-600 to-science-600 opacity-10"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              <span className="block">AI-Powered</span>
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-science-600">
                Raman Analysis
              </span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Advanced DIY Raman spectrometer platform with machine learning-based 
              molecular identification. Professional-grade analysis made accessible.
            </p>
            
            {/* Quick Stats */}
            {stats && !statsLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-2xl mx-auto mb-12"
              >
                <div className="bg-white/80 backdrop-blur rounded-lg p-4 shadow-sm">
                  <div className="text-2xl font-bold text-primary-600">
                    {stats.total_spectra.toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-600">Total Spectra</div>
                </div>
                <div className="bg-white/80 backdrop-blur rounded-lg p-4 shadow-sm">
                  <div className="text-2xl font-bold text-science-600">
                    {stats.unique_compounds.toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-600">Unique Compounds</div>
                </div>
                <div className="bg-white/80 backdrop-blur rounded-lg p-4 shadow-sm">
                  <div className="text-2xl font-bold text-green-600">
                    Ready
                  </div>
                  <div className="text-sm text-gray-600">System Status</div>
                </div>
              </motion.div>
            )}
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="space-x-4"
            >
              <Link href="/analyze" className="btn-primary text-lg px-8 py-3">
                Start Analyzing
              </Link>
              <Link href="/dashboard" className="btn-secondary text-lg px-8 py-3">
                View Dashboard
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="text-center mb-16"
        >
          <motion.h2 variants={itemVariants} className="text-3xl font-bold text-gray-900 mb-4">
            Comprehensive Analysis Platform
          </motion.h2>
          <motion.p variants={itemVariants} className="text-lg text-gray-600 max-w-2xl mx-auto">
            Everything you need for professional Raman spectroscopy analysis, 
            from data acquisition to AI-powered molecular identification.
          </motion.p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          {features.map((feature, index) => (
            <motion.div key={feature.name} variants={itemVariants}>
              <Link href={feature.href} className="group">
                <div className="card h-full hover:shadow-lg transition-all duration-200 group-hover:-translate-y-1">
                  <div className="card-content">
                    <div className={`inline-flex p-3 rounded-lg ${feature.color} text-white mb-4`}>
                      <feature.icon className="h-6 w-6" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-primary-600 transition-colors">
                      {feature.name}
                    </h3>
                    <p className="text-gray-600 text-sm">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Recent Activity Section */}
      <div className="bg-white border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Recent Activity
            </h2>
            <p className="text-lg text-gray-600">
              Latest analysis results and database updates
            </p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Recent Analyses */}
            <div className="card">
              <div className="card-header">
                <h3 className="text-lg font-medium text-gray-900">Recent Analyses</h3>
              </div>
              <div className="card-content">
                <div className="text-center py-8 text-gray-500">
                  <BeakerIcon className="mx-auto h-12 w-12 mb-4 opacity-50" />
                  <p>No recent analyses</p>
                  <Link href="/analyze" className="text-primary-600 hover:text-primary-700 font-medium">
                    Start your first analysis →
                  </Link>
                </div>
              </div>
            </div>
            
            {/* Database Growth */}
            <div className="card">
              <div className="card-header">
                <h3 className="text-lg font-medium text-gray-900">Database Growth</h3>
              </div>
              <div className="card-content">
                <div className="text-center py-8 text-gray-500">
                  <ChartBarIcon className="mx-auto h-12 w-12 mb-4 opacity-50" />
                  <p>Import spectra to see growth trends</p>
                  <Link href="/import" className="text-primary-600 hover:text-primary-700 font-medium">
                    Import data →
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}