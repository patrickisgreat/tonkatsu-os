// Type definitions for Tonkatsu-OS frontend

export interface Spectrum {
  id: string;
  compound_name: string;
  chemical_formula?: string;
  cas_number?: string;
  spectrum_data: number[];
  preprocessed_spectrum?: number[];
  peak_positions?: number[];
  peak_intensities?: number[];
  laser_wavelength: number;
  integration_time: number;
  acquisition_date: string;
  source: string;
  measurement_conditions?: string;
  metadata?: Record<string, any>;
}

export interface AnalysisResult {
  predicted_compound: string;
  confidence: number;
  uncertainty: number;
  model_agreement: number;
  top_predictions: PredictionResult[];
  individual_predictions: IndividualPredictions;
  confidence_analysis: ConfidenceAnalysis;
  processing_time: number;
  method?: string;
  database_matches_found?: number;
  similarity_threshold_used?: number;
  fallback_reason?: string;
  database_match_details?: Record<string, any>;
  external_api_details?: Record<string, any>;
  trained_model_details?: Record<string, any>;
  pretrained_model_details?: Record<string, any>;
}

export interface PredictionResult {
  compound: string;
  probability: number;
}

export interface IndividualPredictions {
  [model: string]: ModelPrediction | undefined;
  random_forest?: ModelPrediction;
  svm?: ModelPrediction;
  neural_network?: ModelPrediction;
  pls_regression?: ModelPrediction;
}

export interface ModelPrediction {
  compound: string;
  confidence: number;
}

export interface ConfidenceAnalysis {
  overall_confidence: number;
  confidence_components: ConfidenceComponents;
  risk_level: 'low' | 'medium' | 'high';
  recommendation: string;
}

export interface ConfidenceComponents {
  [metric: string]: number | undefined;
  probability_score?: number;
  entropy_score?: number;
  peak_match_score?: number;
  model_agreement_score?: number;
  spectral_quality_score?: number;
}

export interface DatabaseStats {
  total_spectra: number;
  unique_compounds: number;
  top_compounds: [string, number][];
}

export interface SimilarSpectrum {
  spectrum_id: string;
  compound_name: string;
  chemical_formula?: string;
  cas_number?: string;
  similarity_score: number;
}

export interface ImportOptions {
  auto_preprocess: boolean;
  validate_data: boolean;
  overwrite_existing: boolean;
  generate_features: boolean;
}

export interface ImportResult {
  total_processed: number;
  successful_integrations: number;
  errors: number;
  error_messages: string[];
}

export interface TrainingResult {
  rf_accuracy: number;
  svm_accuracy: number;
  nn_accuracy: number;
  pls_accuracy?: number;
  ensemble_accuracy: number;
  n_train_samples: number;
  n_val_samples: number;
  n_features: number;
  n_classes: number;
  classification_report: Record<string, any>;
  confusion_matrix: number[][];
  training_time: number;
}

export interface TrainingStatus {
  is_training: boolean;
  progress?: number;
  model_exists: boolean;
  last_trained?: string;
}

export interface SpectralFeatures {
  mean_intensity: number;
  std_intensity: number;
  max_intensity: number;
  min_intensity: number;
  num_peaks: number;
  peak_density: number;
  dominant_peak_position: number;
  dominant_peak_intensity: number;
  spectral_centroid: number;
  spectral_spread: number;
  spectral_skewness: number;
  spectral_kurtosis: number;
}

// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface HardwareStatus {
  connected: boolean;
  port?: string;
  laser_status?: string;
  temperature?: number;
  last_communication?: string;
  last_error?: string;
  last_source?: string;
  last_acquired_at?: string;
  simulate?: boolean;
  simulation_file?: string | null;
  data_points?: number;
}

export interface AcquisitionResponse {
  data: number[];
  source: 'hardware' | 'simulator';
  integration_time: number;
  acquired_at: string;
  port?: string | null;
  simulation_file?: string | null;
}

export interface ReferenceSpectrum {
  id: number;
  compound_name: string;
  chemical_formula?: string;
  cas_number?: string;
  measurement_conditions?: string;
  laser_wavelength?: number;
  integration_time?: number;
  acquisition_date?: string;
  spectrum_data: number[];
  preprocessed_spectrum?: number[] | null;
  metadata?: Record<string, any>;
}

// UI State types
export interface LoadingState {
  isLoading: boolean;
  message?: string;
}

export interface ToastNotification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
}
