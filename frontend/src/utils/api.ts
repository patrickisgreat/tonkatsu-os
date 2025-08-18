// API client for Tonkatsu-OS backend

import axios from 'axios'
import { 
  Spectrum, 
  AnalysisResult, 
  DatabaseStats, 
  SimilarSpectrum,
  ImportResult,
  TrainingResult,
  ApiResponse 
} from '@/types/spectrum'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API Error:', error)
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error)
    }
    throw error
  }
)

export const api = {
  // Database operations
  async getDatabaseStats(): Promise<DatabaseStats> {
    return apiClient.get('/database/stats')
  },

  async getSpectrum(id: string): Promise<Spectrum> {
    return apiClient.get(`/database/spectrum/${id}`)
  },

  async searchSpectra(query: string): Promise<Spectrum[]> {
    return apiClient.get(`/database/search?q=${encodeURIComponent(query)}`)
  },

  async getSimilarSpectra(spectrumData: number[], topK: number = 5): Promise<SimilarSpectrum[]> {
    return apiClient.post('/database/similar', {
      spectrum_data: spectrumData,
      top_k: topK
    })
  },

  // Analysis operations
  async analyzeSpectrum(spectrumData: number[]): Promise<AnalysisResult> {
    return apiClient.post('/analysis/analyze', {
      spectrum_data: spectrumData
    })
  },

  async preprocessSpectrum(spectrumData: number[], options?: any): Promise<number[]> {
    return apiClient.post('/analysis/preprocess', {
      spectrum_data: spectrumData,
      options
    })
  },

  // Import operations
  async importSpectrum(file: File, metadata: any): Promise<ImportResult> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('metadata', JSON.stringify(metadata))
    
    return apiClient.post('/import/spectrum', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },

  async importBatch(files: File[], options: any): Promise<ImportResult> {
    const formData = new FormData()
    files.forEach((file, index) => {
      formData.append(`files`, file)
    })
    formData.append('options', JSON.stringify(options))
    
    return apiClient.post('/import/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },

  // Public database integration
  async downloadRRUFFData(maxSpectra: number = 50): Promise<ImportResult> {
    return apiClient.post('/data/rruff/download', {
      max_spectra: maxSpectra
    })
  },

  async generateSyntheticData(samplesPerCompound: number = 10): Promise<ImportResult> {
    return apiClient.post('/data/synthetic/generate', {
      samples_per_compound: samplesPerCompound
    })
  },

  async getRRUFFStatus(): Promise<{ available: boolean; last_update?: string; count: number }> {
    return apiClient.get('/data/rruff/status')
  },

  // Training operations
  async trainModels(config?: any): Promise<TrainingResult> {
    return apiClient.post('/training/train', config || {})
  },

  async getTrainingStatus(): Promise<{ 
    is_training: boolean; 
    progress?: number; 
    model_exists: boolean 
  }> {
    return apiClient.get('/training/status')
  },

  async getModelMetrics(): Promise<any> {
    return apiClient.get('/training/metrics')
  },

  // Acquisition (hardware interface)
  async acquireSpectrum(integrationTime?: number): Promise<number[]> {
    return apiClient.post('/acquisition/acquire', {
      integration_time: integrationTime || 200
    })
  },

  async getHardwareStatus(): Promise<{
    connected: boolean;
    port?: string;
    laser_status?: string;
  }> {
    return apiClient.get('/acquisition/status')
  },

  // System operations
  async getSystemHealth(): Promise<{
    status: 'healthy' | 'warning' | 'error';
    components: Record<string, boolean>;
    version: string;
  }> {
    return apiClient.get('/system/health')
  },

  async exportDatabase(format: 'csv' | 'json' | 'sqlite'): Promise<Blob> {
    const response = await apiClient.get(`/system/export/${format}`, {
      responseType: 'blob'
    })
    return response.data
  },

  // Utility functions
  async uploadFile(file: File, endpoint: string): Promise<any> {
    const formData = new FormData()
    formData.append('file', file)
    
    return apiClient.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },
}

export default api