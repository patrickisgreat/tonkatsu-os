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
}

export interface PredictionResult {
  compound: string;
  probability: number;
}

export interface IndividualPredictions {
  random_forest: ModelPrediction;
  svm: ModelPrediction;
  neural_network: ModelPrediction;
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
  probability_score: number;
  entropy_score: number;
  peak_match_score: number;
  model_agreement_score: number;
  spectral_quality_score: number;
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
  ensemble_accuracy: number;
  n_train_samples: number;
  n_val_samples: number;
  n_features: number;
  n_classes: number;
  classification_report: Record<string, any>;
  confusion_matrix: number[][];
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