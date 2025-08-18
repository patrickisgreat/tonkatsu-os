'use client'

import React from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions,
  ChartData,
  ScriptableContext
} from 'chart.js'
import { Line } from 'react-chartjs-2'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

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
  backgroundColor = 'rgba(59, 130, 246, 0.1)'
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

  const chartData: ChartData<'line'> = {
    labels: xAxisData.map(x => x.toFixed(0)),
    datasets: [
      {
        label: compoundName,
        data: spectrumData,
        borderColor: color,
        backgroundColor: backgroundColor,
        borderWidth: 2,
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointBackgroundColor: color,
        pointBorderColor: '#ffffff',
        pointBorderWidth: 2,
      },
      // Peak markers
      ...(showPeaks && peaks.length > 0 ? [{
        label: 'Peaks',
        data: peaks.map(peakIdx => ({
          x: xAxisData[peakIdx].toFixed(0),
          y: spectrumData[peakIdx]
        })),
        borderColor: 'rgb(239, 68, 68)', // red-500
        backgroundColor: 'rgba(239, 68, 68, 0.8)',
        borderWidth: 0,
        pointRadius: 6,
        pointHoverRadius: 8,
        showLine: false,
        pointStyle: 'triangle' as const,
      }] : [])
    ]
  }

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          font: {
            size: 12,
            family: 'Inter, sans-serif'
          },
          color: '#374151', // gray-700
          usePointStyle: true,
        }
      },
      title: {
        display: true,
        text: `Raman Spectrum - ${compoundName}`,
        font: {
          size: 16,
          weight: 'bold',
          family: 'Inter, sans-serif'
        },
        color: '#111827', // gray-900
        padding: 20
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: color,
        borderWidth: 1,
        cornerRadius: 6,
        titleFont: {
          size: 12,
          weight: 'bold'
        },
        bodyFont: {
          size: 11
        },
        callbacks: {
          title: (context) => {
            const dataIndex = context[0]?.dataIndex ?? 0
            return `Wavenumber: ${xAxisData[dataIndex]?.toFixed(1) ?? ''} cm⁻¹`
          },
          label: (context) => {
            return `Intensity: ${context.parsed.y.toFixed(3)}`
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Raman Shift (cm⁻¹)',
          font: {
            size: 14,
            weight: 'bold',
            family: 'Inter, sans-serif'
          },
          color: '#374151' // gray-700
        },
        grid: {
          display: true,
          color: 'rgba(156, 163, 175, 0.2)', // gray-400 with opacity
          drawOnChartArea: true,
          drawTicks: true,
        },
        ticks: {
          color: '#6B7280', // gray-500
          font: {
            size: 11,
            family: 'Inter, sans-serif'
          },
          maxTicksLimit: 10,
          callback: function(value) {
            const numValue = Number(value)
            return numValue % 500 === 0 ? numValue.toString() : ''
          }
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Intensity (a.u.)',
          font: {
            size: 14,
            weight: 'bold',
            family: 'Inter, sans-serif'
          },
          color: '#374151' // gray-700
        },
        grid: {
          display: true,
          color: 'rgba(156, 163, 175, 0.2)', // gray-400 with opacity
          drawOnChartArea: true,
        },
        ticks: {
          color: '#6B7280', // gray-500
          font: {
            size: 11,
            family: 'Inter, sans-serif'
          },
          callback: function(value) {
            return Number(value).toFixed(2)
          }
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    },
    elements: {
      point: {
        hoverRadius: 6,
        hitRadius: 6
      }
    },
    animation: {
      duration: 750,
      easing: 'easeInOutQuart'
    }
  }

  return (
    <div className="w-full" style={{ height: `${height}px` }}>
      <Line data={chartData} options={chartOptions} />
      
      {/* Spectrum Statistics */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="font-medium text-gray-700">Data Points</div>
          <div className="text-lg font-bold text-gray-900">{spectrumData.length}</div>
        </div>
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="font-medium text-gray-700">Max Intensity</div>
          <div className="text-lg font-bold text-gray-900">{Math.max(...spectrumData).toFixed(3)}</div>
        </div>
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="font-medium text-gray-700">Mean Intensity</div>
          <div className="text-lg font-bold text-gray-900">
            {(spectrumData.reduce((a, b) => a + b, 0) / spectrumData.length).toFixed(3)}
          </div>
        </div>
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="font-medium text-gray-700">Peaks Found</div>
          <div className="text-lg font-bold text-gray-900">{peaks.length}</div>
        </div>
      </div>

      {/* Peak List */}
      {showPeaks && peaks.length > 0 && (
        <div className="mt-4 bg-white border rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-2">Identified Peaks</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2 text-sm">
            {peaks.slice(0, 10).map((peakIdx, index) => (
              <div key={index} className="bg-red-50 p-2 rounded text-center">
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