'use client'

import { SpectralChart } from '@/components/SpectralChart'
import type { ReactNode } from 'react'

interface SpectrumPreviewProps {
  title: string
  subtitle?: string
  spectrumData?: number[]
  loading?: boolean
  emptyMessage?: string
  color?: string
  backgroundColor?: string
  showPeaks?: boolean
  footer?: ReactNode
  className?: string
}

export function SpectrumPreview({
  title,
  subtitle,
  spectrumData,
  loading = false,
  emptyMessage = 'No spectrum available.',
  color = 'rgb(59, 130, 246)',
  backgroundColor = 'rgba(59, 130, 246, 0.08)',
  showPeaks = false,
  footer,
  className,
}: SpectrumPreviewProps) {
  const containerClass = className ? `card ${className}` : 'card'

  return (
    <div className={containerClass}>
      <div className="card-header">
        <div>
          <h3 className="text-lg font-medium text-gray-900">{title}</h3>
          {subtitle && <p className="text-sm text-gray-600 mt-1">{subtitle}</p>}
        </div>
      </div>
      <div className="card-content">
        {loading ? (
          <div className="flex items-center justify-center py-16 text-sm text-gray-500">
            Loading spectrum...
          </div>
        ) : spectrumData && spectrumData.length > 0 ? (
          <SpectralChart
            spectrumData={spectrumData}
            showPeaks={showPeaks}
            height={280}
            color={color}
            backgroundColor={backgroundColor}
          />
        ) : (
          <div className="flex items-center justify-center py-16 text-sm text-gray-500">
            {emptyMessage}
          </div>
        )}
      </div>
      {footer && (
        <div className="border-t border-gray-100 px-6 py-4 text-sm text-gray-600">
          {footer}
        </div>
      )}
    </div>
  )
}
