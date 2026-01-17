'use client'

import React from 'react'
import dynamic from 'next/dynamic'
import type { Data, Layout, Config } from 'plotly.js'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(
  async () => {
    const Plotly = await import('plotly.js-dist-min')
    const createPlotlyComponent = (await import('react-plotly.js/factory')).default
    return createPlotlyComponent(Plotly.default || Plotly)
  },
  { ssr: false }
) as React.ComponentType<{
  data: Data[]
  layout: Partial<Layout>
  config: Partial<Config>
  style?: React.CSSProperties
  useResizeHandler?: boolean
}>

interface SpectralChartProps {
  spectrumData: number[]
  compoundName?: string
  wavelengthRange?: [number, number]
  showPeaks?: boolean
  height?: number
  color?: string
  backgroundColor?: string
}

export const SpectralChart: React.FC<SpectralChartProps> = ({
  spectrumData,
  compoundName = 'Unknown Compound',
  wavelengthRange = [400, 4000],
  showPeaks = false,
  height = 400,
  color = 'rgb(59, 130, 246)', // blue-500
  backgroundColor = 'rgba(59, 130, 246, 0.2)'
}) => {
  // Generate wavelength/wavenumber axis
  const generateXAxis = (dataLength: number, range: [number, number]): number[] => {
    const [min, max] = range
    const step = (max - min) / (dataLength - 1)
    return Array.from({ length: dataLength }, (_, i) => min + (i * step))
  }

  // Find peaks in the spectrum
  const findPeaks = (data: number[], threshold: number = 0.1): number[] => {
    const peaks: number[] = []
    const maxVal = Math.max(...data)
    const minThreshold = maxVal * threshold

    for (let i = 1; i < data.length - 1; i++) {
      if (data[i] > data[i - 1] &&
          data[i] > data[i + 1] &&
          data[i] > minThreshold) {
        peaks.push(i)
      }
    }

    // Return top 10 peaks sorted by intensity
    return peaks
      .sort((a, b) => data[b] - data[a])
      .slice(0, 10)
  }

  const xAxisData = generateXAxis(spectrumData.length, wavelengthRange)
  const peaks = showPeaks ? findPeaks(spectrumData) : []

  // Plotly trace for main spectrum
  const spectrumTrace: Data = {
    x: xAxisData,
    y: spectrumData,
    type: 'scatter',
    mode: 'lines',
    name: compoundName,
    line: {
      color: color,
      width: 2,
    },
    fill: 'tozeroy',
    fillcolor: backgroundColor,
    hovertemplate: 'Wavenumber: %{x:.1f} cm⁻¹<br>Intensity: %{y:.3f}<extra></extra>',
  }

  // Plotly trace for peaks
  const peaksTrace: Data = {
    x: peaks.map(i => xAxisData[i]),
    y: peaks.map(i => spectrumData[i]),
    type: 'scatter',
    mode: 'markers',
    name: 'Peaks',
    marker: {
      color: 'rgb(239, 68, 68)',
      size: 12,
      symbol: 'triangle-up',
      line: {
        color: 'white',
        width: 1,
      },
    },
    hovertemplate: 'Peak: %{x:.0f} cm⁻¹<br>Intensity: %{y:.3f}<extra></extra>',
  }

  const traces: Data[] = showPeaks && peaks.length > 0
    ? [spectrumTrace, peaksTrace]
    : [spectrumTrace]

  const layout: Partial<Layout> = {
    title: {
      text: `Raman Spectrum - ${compoundName}`,
      font: {
        size: 16,
        color: '#111827',
        family: 'Inter, sans-serif',
      },
    },
    xaxis: {
      title: {
        text: 'Raman Shift (cm⁻¹)',
        font: {
          size: 14,
          color: '#374151',
          family: 'Inter, sans-serif',
        },
      },
      tickfont: {
        size: 11,
        color: '#6B7280',
      },
      gridcolor: 'rgba(156, 163, 175, 0.2)',
      zeroline: false,
    },
    yaxis: {
      title: {
        text: 'Intensity (a.u.)',
        font: {
          size: 14,
          color: '#374151',
          family: 'Inter, sans-serif',
        },
      },
      tickfont: {
        size: 11,
        color: '#6B7280',
      },
      gridcolor: 'rgba(156, 163, 175, 0.2)',
      zeroline: false,
    },
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'right',
      x: 1,
      font: {
        size: 12,
        color: '#374151',
      },
    },
    margin: {
      l: 60,
      r: 30,
      t: 60,
      b: 50,
    },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    hovermode: 'x unified',
    dragmode: 'zoom',
  }

  const config: Partial<Config> = {
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false,
    responsive: true,
    scrollZoom: true,
    toImageButtonOptions: {
      format: 'png',
      filename: `raman_spectrum_${compoundName.replace(/\s+/g, '_')}`,
      height: 800,
      width: 1200,
      scale: 2,
    },
  }

  return (
    <div className="w-full flex flex-col gap-4">
      {/* Interactive Plotly Chart */}
      <div className="w-full bg-white rounded-lg border shadow-sm overflow-hidden">
        <Plot
          data={traces}
          layout={layout}
          config={config}
          style={{ width: '100%', height: `${height}px` }}
          useResizeHandler={true}
        />
      </div>

      {/* Spectrum Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="bg-white border p-3 rounded-lg shadow-sm">
          <div className="font-medium text-gray-700">Data Points</div>
          <div className="text-lg font-bold text-gray-900">{spectrumData.length}</div>
        </div>
        <div className="bg-white border p-3 rounded-lg shadow-sm">
          <div className="font-medium text-gray-700">Max Intensity</div>
          <div className="text-lg font-bold text-gray-900">{Math.max(...spectrumData).toFixed(3)}</div>
        </div>
        <div className="bg-white border p-3 rounded-lg shadow-sm">
          <div className="font-medium text-gray-700">Mean Intensity</div>
          <div className="text-lg font-bold text-gray-900">
            {(spectrumData.reduce((a, b) => a + b, 0) / spectrumData.length).toFixed(3)}
          </div>
        </div>
        <div className="bg-white border p-3 rounded-lg shadow-sm">
          <div className="font-medium text-gray-700">Peaks Found</div>
          <div className="text-lg font-bold text-gray-900">{peaks.length}</div>
        </div>
      </div>

      {/* Peak List */}
      {showPeaks && peaks.length > 0 && (
        <div className="bg-white border rounded-lg p-4 shadow-sm">
          <h4 className="font-medium text-gray-900 mb-3">Identified Peaks</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2 text-sm">
            {peaks.slice(0, 10).map((peakIdx, index) => (
              <div key={index} className="bg-red-50 border border-red-200 p-2 rounded text-center">
                <div className="font-medium text-red-800">
                  {xAxisData[peakIdx].toFixed(0)} cm⁻¹
                </div>
                <div className="text-red-600 text-xs">
                  {spectrumData[peakIdx].toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default SpectralChart
