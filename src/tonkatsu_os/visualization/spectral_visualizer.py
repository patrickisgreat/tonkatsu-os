"""
Advanced Visualization Module for Raman Spectral Analysis

This module provides beautiful, interactive visualizations for spectral data,
analysis results, and database exploration using Plotly and Streamlit.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from tonkatsu_os.database import RamanSpectralDatabase
from tonkatsu_os.preprocessing import AdvancedPreprocessor
import logging

logger = logging.getLogger(__name__)


class SpectralVisualizer:
    """
    Comprehensive visualization system for Raman spectral analysis
    with interactive plots and professional reporting.
    """
    
    def __init__(self, database: RamanSpectralDatabase):
        self.database = database
        self.preprocessor = AdvancedPreprocessor()
        
        # Color schemes for different plot types
        self.color_schemes = {
            'default': px.colors.qualitative.Set1,
            'scientific': px.colors.sequential.Viridis,
            'peaks': px.colors.qualitative.Dark24,
            'confidence': ['#FF6B6B', '#FFE66D', '#4ECDC4', '#45B7D1', '#96CEB4']
        }
    
    def create_visualization_dashboard(self):
        """Create the main visualization dashboard."""
        st.header("📊 Spectral Visualization Dashboard")
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Single Spectrum", "📈 Multi-Spectrum", "🎯 Analysis Results", 
            "📊 Database Explorer", "📋 Reports"
        ])
        
        with tab1:
            self._single_spectrum_tab()
        
        with tab2:
            self._multi_spectrum_tab()
        
        with tab3:
            self._analysis_results_tab()
        
        with tab4:
            self._database_explorer_tab()
        
        with tab5:
            self._reports_tab()
    
    def _single_spectrum_tab(self):
        """Single spectrum visualization with detailed analysis."""
        st.subheader("🔍 Single Spectrum Analysis")
        
        # Spectrum selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Get available spectra
            stats = self.database.get_database_stats()
            
            if stats['total_spectra'] == 0:
                st.warning("No spectra in database. Import some data first!")
                return
            
            # Compound selection
            compounds = [comp[0] for comp in stats['top_compounds']]
            selected_compound = st.selectbox("Select compound:", compounds)
            
            # Get spectra for selected compound
            spectra_list = self.database.search_by_compound_name(selected_compound, exact_match=True)
            
            if not spectra_list:
                st.error("No spectra found for selected compound")
                return
            
            spectrum_ids = [f"ID {spec['id']}" for spec in spectra_list]
            selected_spectrum_id = st.selectbox("Select spectrum:", spectrum_ids)
            
            # Analysis options
            st.subheader("Analysis Options")
            show_peaks = st.checkbox("Show peaks", value=True)
            show_baseline = st.checkbox("Show baseline", value=False)
            normalize_display = st.checkbox("Normalize for display", value=True)
            
            # Processing options
            st.subheader("Processing Options")
            apply_smoothing = st.checkbox("Apply smoothing", value=False)
            remove_background = st.checkbox("Remove background", value=False)
        
        with col2:
            # Get and display spectrum
            spectrum_id = int(selected_spectrum_id.split()[1])
            spectrum_data = self.database.get_spectrum_by_id(spectrum_id)
            
            if spectrum_data:
                self._plot_single_spectrum_detailed(
                    spectrum_data, show_peaks, show_baseline, 
                    normalize_display, apply_smoothing, remove_background
                )
    
    def _plot_single_spectrum_detailed(self, spectrum_data: Dict, show_peaks: bool,
                                     show_baseline: bool, normalize_display: bool,
                                     apply_smoothing: bool, remove_background: bool):
        """Create detailed single spectrum plot with analysis."""
        
        # Get spectrum array
        if spectrum_data['preprocessed_spectrum'] is not None:
            spectrum = spectrum_data['preprocessed_spectrum']
        else:
            spectrum = spectrum_data['spectrum_data']
        
        # Apply additional processing if requested
        processed_spectrum = spectrum.copy()
        
        if apply_smoothing:
            processed_spectrum = self.preprocessor.smooth_spectrum(processed_spectrum)
        
        if remove_background:
            processed_spectrum = self.preprocessor.baseline_correction(processed_spectrum)
        
        if normalize_display:
            processed_spectrum = self.preprocessor.normalize_spectrum(processed_spectrum)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Main Spectrum', 'Peak Analysis', 'Statistics', 'Quality Metrics'),
            specs=[[{"colspan": 2}, None],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Main spectrum plot
        x_axis = np.arange(len(processed_spectrum))
        
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=processed_spectrum,
                mode='lines',
                name='Spectrum',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Point: %{x}<br>Intensity: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add peaks if requested
        if show_peaks and spectrum_data['peak_positions'] is not None:
            peaks = spectrum_data['peak_positions']
            peak_intensities = spectrum_data['peak_intensities']
            
            # Filter peaks that are within the current spectrum range
            valid_peaks = peaks < len(processed_spectrum)
            peaks = peaks[valid_peaks]
            peak_intensities = processed_spectrum[peaks]
            
            fig.add_trace(
                go.Scatter(
                    x=peaks, y=peak_intensities,
                    mode='markers',
                    name='Peaks',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='diamond'
                    ),
                    hovertemplate='Peak at: %{x}<br>Intensity: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Annotate top 5 peaks
            for i, (peak, intensity) in enumerate(zip(peaks[:5], peak_intensities[:5])):
                fig.add_annotation(
                    x=peak, y=intensity,
                    text=f"{peak}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="red",
                    row=1, col=1
                )
        
        # Peak analysis histogram
        if spectrum_data['peak_positions'] is not None:
            peaks = spectrum_data['peak_positions']
            peak_intensities = spectrum_data['peak_intensities']
            
            # Create peak intensity distribution
            fig.add_trace(
                go.Bar(
                    x=[f"Peak {i+1}" for i in range(min(10, len(peaks)))],
                    y=peak_intensities[:10],
                    name='Top 10 Peaks',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # Statistics table
        stats_data = self._calculate_spectrum_statistics(processed_spectrum, spectrum_data)
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                          fill_color='lightblue',
                          align='center'),
                cells=dict(values=[list(stats_data.keys()), list(stats_data.values())],
                          fill_color='white',
                          align='center')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Detailed Analysis: {spectrum_data['compound_name']}",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Data Point", row=1, col=1)
        fig.update_yaxes(title_text="Intensity", row=1, col=1)
        fig.update_xaxes(title_text="Peak Number", row=2, col=1)
        fig.update_yaxes(title_text="Intensity", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis
        self._display_spectrum_metadata(spectrum_data)
    
    def _multi_spectrum_tab(self):
        """Multi-spectrum comparison visualization."""
        st.subheader("📈 Multi-Spectrum Comparison")
        
        # Spectrum selection interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Spectra")
            
            # Get all compounds
            stats = self.database.get_database_stats()
            compounds = [comp[0] for comp in stats['top_compounds']]
            
            selected_compounds = st.multiselect(
                "Choose compounds to compare:",
                compounds,
                default=compounds[:min(3, len(compounds))]
            )
            
            if not selected_compounds:
                st.warning("Please select at least one compound")
                return
            
            # Comparison options
            st.subheader("Comparison Options")
            overlay_type = st.radio(
                "Display type:",
                ["Overlay", "Stacked", "Normalized"]
            )
            
            show_peaks_multi = st.checkbox("Show all peaks", key="multi_peaks")
            show_differences = st.checkbox("Highlight differences")
            
            # Color scheme
            color_scheme = st.selectbox(
                "Color scheme:",
                ["Default", "Scientific", "Peaks"]
            ).lower()
        
        with col2:
            # Get spectra for selected compounds
            all_spectra = []
            for compound in selected_compounds:
                spectra_list = self.database.search_by_compound_name(compound, exact_match=True)
                if spectra_list:
                    # Take first spectrum for each compound
                    spectrum_data = self.database.get_spectrum_by_id(spectra_list[0]['id'])
                    if spectrum_data:
                        all_spectra.append(spectrum_data)
            
            if all_spectra:
                self._plot_multi_spectrum_comparison(
                    all_spectra, overlay_type, show_peaks_multi, 
                    show_differences, color_scheme
                )
    
    def _plot_multi_spectrum_comparison(self, spectra_list: List[Dict], 
                                       overlay_type: str, show_peaks: bool,
                                       show_differences: bool, color_scheme: str):
        """Create multi-spectrum comparison plot."""
        
        fig = go.Figure()
        
        colors = self.color_schemes.get(color_scheme, self.color_schemes['default'])
        
        # Normalize all spectra to same length
        max_length = max(len(spec['preprocessed_spectrum'] or spec['spectrum_data']) 
                        for spec in spectra_list)
        
        processed_spectra = []
        for i, spectrum_data in enumerate(spectra_list):
            # Get spectrum
            if spectrum_data['preprocessed_spectrum'] is not None:
                spectrum = spectrum_data['preprocessed_spectrum']
            else:
                spectrum = spectrum_data['spectrum_data']
            
            # Pad or truncate to max_length
            if len(spectrum) < max_length:
                spectrum = np.pad(spectrum, (0, max_length - len(spectrum)), 'constant')
            else:
                spectrum = spectrum[:max_length]
            
            # Apply normalization if requested
            if overlay_type == "Normalized":
                spectrum = self.preprocessor.normalize_spectrum(spectrum)
            elif overlay_type == "Stacked":
                spectrum = spectrum + i * (np.max(spectrum) * 1.2)
            
            processed_spectra.append(spectrum)
            
            # Plot spectrum
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(spectrum)),
                    y=spectrum,
                    mode='lines',
                    name=spectrum_data['compound_name'],
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"{spectrum_data['compound_name']}<br>" +
                                "Point: %{x}<br>Intensity: %{y:.3f}<extra></extra>"
                )
            )
            
            # Add peaks if requested
            if show_peaks and spectrum_data['peak_positions'] is not None:
                peaks = spectrum_data['peak_positions']
                valid_peaks = peaks < len(spectrum)
                peaks = peaks[valid_peaks]
                peak_intensities = spectrum[peaks]
                
                fig.add_trace(
                    go.Scatter(
                        x=peaks,
                        y=peak_intensities,
                        mode='markers',
                        name=f"{spectrum_data['compound_name']} Peaks",
                        marker=dict(
                            color=colors[i % len(colors)],
                            size=6,
                            symbol='diamond'
                        ),
                        showlegend=False
                    )
                )
        
        # Add difference highlighting if requested
        if show_differences and len(processed_spectra) >= 2:
            # Calculate standard deviation across spectra
            spectrum_array = np.array(processed_spectra)
            std_dev = np.std(spectrum_array, axis=0)
            mean_spectrum = np.mean(spectrum_array, axis=0)
            
            # Highlight regions with high variation
            high_var_threshold = np.percentile(std_dev, 75)
            high_var_regions = std_dev > high_var_threshold
            
            if np.any(high_var_regions):
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(mean_spectrum))[high_var_regions],
                        y=mean_spectrum[high_var_regions],
                        mode='markers',
                        name='High Variation Regions',
                        marker=dict(
                            color='orange',
                            size=4,
                            symbol='x'
                        )
                    )
                )
        
        # Update layout
        title = f"Multi-Spectrum Comparison ({overlay_type})"
        fig.update_layout(
            title=title,
            xaxis_title="Data Point",
            yaxis_title="Intensity",
            height=600,
            template="plotly_white",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison statistics
        self._display_comparison_statistics(processed_spectra, spectra_list)
    
    def _analysis_results_tab(self):
        """Display analysis results with confidence visualization."""
        st.subheader("🎯 Analysis Results Visualization")
        
        # Check if we have analysis results in session state
        if 'current_spectrum' not in st.session_state:
            st.info("No analysis results available. Analyze a spectrum first!")
            return
        
        spectrum_info = st.session_state.current_spectrum
        
        # Mock analysis results for demonstration
        # In real implementation, this would come from actual ML analysis
        mock_results = {
            'predicted_compound': 'Benzene',
            'confidence': 0.87,
            'uncertainty': 0.13,
            'model_agreement': 0.92,
            'top_predictions': [
                {'compound': 'Benzene', 'probability': 0.87},
                {'compound': 'Toluene', 'probability': 0.08},
                {'compound': 'Xylene', 'probability': 0.05}
            ],
            'individual_predictions': {
                'random_forest': {'compound': 'Benzene', 'confidence': 0.89},
                'svm': {'compound': 'Benzene', 'confidence': 0.85},
                'neural_network': {'compound': 'Benzene', 'confidence': 0.87}
            }
        }
        
        mock_confidence_analysis = {
            'overall_confidence': 0.87,
            'confidence_components': {
                'probability_score': 0.87,
                'entropy_score': 0.92,
                'peak_match_score': 0.78,
                'model_agreement_score': 0.92,
                'spectral_quality_score': 0.95
            },
            'risk_level': 'low',
            'recommendation': 'High confidence identification. Results can be trusted.'
        }
        
        self._plot_analysis_results(mock_results, mock_confidence_analysis, spectrum_info)
    
    def _plot_analysis_results(self, results: Dict, confidence_analysis: Dict, spectrum_info: Dict):
        """Create comprehensive analysis results visualization."""
        
        # Create multi-panel results display
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence breakdown pie chart
            fig_confidence = go.Figure(data=[go.Pie(
                labels=list(confidence_analysis['confidence_components'].keys()),
                values=list(confidence_analysis['confidence_components'].values()),
                hole=.3,
                marker_colors=self.color_schemes['confidence']
            )])
            
            fig_confidence.update_layout(
                title="Confidence Component Breakdown",
                height=400,
                annotations=[dict(text=f"{confidence_analysis['overall_confidence']:.1%}<br>Overall", 
                                x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        with col2:
            # Top predictions bar chart
            compounds = [pred['compound'] for pred in results['top_predictions']]
            probabilities = [pred['probability'] for pred in results['top_predictions']]
            
            fig_predictions = go.Figure(data=[
                go.Bar(
                    x=probabilities,
                    y=compounds,
                    orientation='h',
                    marker_color=px.colors.sequential.Blues_r[:len(compounds)]
                )
            ])
            
            fig_predictions.update_layout(
                title="Top Predictions",
                xaxis_title="Probability",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_predictions, use_container_width=True)
        
        # Model agreement visualization
        st.subheader("Model Agreement Analysis")
        
        models = list(results['individual_predictions'].keys())
        model_compounds = [results['individual_predictions'][model]['compound'] for model in models]
        model_confidences = [results['individual_predictions'][model]['confidence'] for model in models]
        
        fig_models = go.Figure()
        
        # Add bars for each model
        fig_models.add_trace(go.Bar(
            x=models,
            y=model_confidences,
            text=model_compounds,
            textposition='auto',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
        ))
        
        # Add ensemble result line
        fig_models.add_shape(
            type="line",
            x0=-0.5, y0=results['confidence'], x1=len(models)-0.5, y1=results['confidence'],
            line=dict(color="red", width=3, dash="dash"),
        )
        
        fig_models.add_annotation(
            x=len(models)-1, y=results['confidence'],
            text=f"Ensemble: {results['confidence']:.1%}",
            showarrow=True,
            arrowhead=2
        )
        
        fig_models.update_layout(
            title="Individual Model vs Ensemble Performance",
            xaxis_title="Model",
            yaxis_title="Confidence",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_models, use_container_width=True)
        
        # Risk assessment visualization
        self._plot_risk_assessment(confidence_analysis)
        
        # Spectrum with prediction overlay
        st.subheader("Analyzed Spectrum")
        self._plot_spectrum_with_prediction_overlay(spectrum_info, results)
    
    def _plot_risk_assessment(self, confidence_analysis: Dict):
        """Create risk assessment visualization."""
        st.subheader("Risk Assessment")
        
        risk_level = confidence_analysis['risk_level']
        overall_confidence = confidence_analysis['overall_confidence']
        
        # Risk gauge chart
        fig_risk = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Level"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_risk.update_layout(height=300)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            # Risk level indicator
            risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
            st.markdown(f"### Risk Level: {risk_colors[risk_level]} {risk_level.upper()}")
            
            st.markdown(f"**Recommendation:**")
            st.info(confidence_analysis['recommendation'])
            
            # Component scores
            st.markdown("**Component Scores:**")
            for component, score in confidence_analysis['confidence_components'].items():
                score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                st.text(f"{component.replace('_', ' ').title()}: {score_bar} {score:.2f}")
    
    def _plot_spectrum_with_prediction_overlay(self, spectrum_info: Dict, results: Dict):
        """Plot spectrum with prediction information overlay."""
        
        spectrum = spectrum_info['processed']
        peaks = spectrum_info['peaks']
        peak_intensities = spectrum_info['peak_intensities']
        
        fig = go.Figure()
        
        # Main spectrum
        fig.add_trace(go.Scatter(
            x=np.arange(len(spectrum)),
            y=spectrum,
            mode='lines',
            name='Analyzed Spectrum',
            line=dict(color='blue', width=2)
        ))
        
        # Peaks
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=peaks,
                y=peak_intensities,
                mode='markers',
                name='Detected Peaks',
                marker=dict(color='red', size=8, symbol='diamond')
            ))
        
        # Add prediction annotation
        fig.add_annotation(
            x=len(spectrum) * 0.7,
            y=np.max(spectrum) * 0.9,
            text=f"Predicted: {results['predicted_compound']}<br>" +
                 f"Confidence: {results['confidence']:.1%}<br>" +
                 f"Model Agreement: {results['model_agreement']:.1%}",
            showarrow=True,
            arrowhead=2,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title="Spectrum Analysis Results",
            xaxis_title="Data Point",
            yaxis_title="Normalized Intensity",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _database_explorer_tab(self):
        """Database exploration and statistics visualization."""
        st.subheader("📊 Database Explorer")
        
        stats = self.database.get_database_stats()
        
        if stats['total_spectra'] == 0:
            st.warning("Database is empty. Import some spectra first!")
            return
        
        # Database overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Spectra", stats['total_spectra'])
        
        with col2:
            st.metric("Unique Compounds", stats['unique_compounds'])
        
        with col3:
            # Calculate average spectra per compound
            avg_spectra = stats['total_spectra'] / max(stats['unique_compounds'], 1)
            st.metric("Avg Spectra/Compound", f"{avg_spectra:.1f}")
        
        with col4:
            # Database size (mock calculation)
            db_size = stats['total_spectra'] * 0.1  # Assume ~0.1 MB per spectrum
            st.metric("Est. Database Size", f"{db_size:.1f} MB")
        
        # Compound distribution
        st.subheader("Compound Distribution")
        
        compound_data = pd.DataFrame(stats['top_compounds'], columns=['Compound', 'Count'])
        
        fig_dist = px.bar(
            compound_data.head(15),  # Show top 15
            x='Compound',
            y='Count',
            title="Top Compounds by Number of Spectra",
            color='Count',
            color_continuous_scale='Blues'
        )
        
        fig_dist.update_layout(
            xaxis_tickangle=-45,
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Interactive compound explorer
        st.subheader("Interactive Compound Explorer")
        
        selected_compound = st.selectbox(
            "Select compound to explore:",
            [comp[0] for comp in stats['top_compounds']]
        )
        
        if selected_compound:
            self._explore_compound_details(selected_compound)
    
    def _explore_compound_details(self, compound_name: str):
        """Explore details for a specific compound."""
        # Get all spectra for this compound
        spectra_list = self.database.search_by_compound_name(compound_name, exact_match=True)
        
        st.write(f"**{compound_name}** - {len(spectra_list)} spectra available")
        
        if len(spectra_list) > 1:
            # Show variation between different measurements
            st.subheader("Measurement Variation")
            
            # Get a few representative spectra
            sample_spectra = []
            for i, spec_info in enumerate(spectra_list[:5]):  # Max 5 spectra
                spectrum_data = self.database.get_spectrum_by_id(spec_info['id'])
                if spectrum_data:
                    sample_spectra.append({
                        'name': f"Measurement {i+1}",
                        'data': spectrum_data['preprocessed_spectrum'] or spectrum_data['spectrum_data'],
                        'id': spec_info['id']
                    })
            
            if sample_spectra:
                fig_variation = go.Figure()
                
                for spec in sample_spectra:
                    fig_variation.add_trace(go.Scatter(
                        x=np.arange(len(spec['data'])),
                        y=spec['data'],
                        mode='lines',
                        name=spec['name'],
                        opacity=0.7
                    ))
                
                fig_variation.update_layout(
                    title=f"Spectral Variation for {compound_name}",
                    xaxis_title="Data Point",
                    yaxis_title="Intensity",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_variation, use_container_width=True)
                
                # Calculate and display variation statistics
                if len(sample_spectra) > 1:
                    self._calculate_variation_statistics(sample_spectra, compound_name)
    
    def _reports_tab(self):
        """Generate and display analysis reports."""
        st.subheader("📋 Analysis Reports")
        
        report_type = st.selectbox(
            "Select report type:",
            ["Database Summary", "Spectrum Analysis", "Quality Assessment", "Custom Report"]
        )
        
        if report_type == "Database Summary":
            self._generate_database_summary_report()
        elif report_type == "Spectrum Analysis":
            self._generate_spectrum_analysis_report()
        elif report_type == "Quality Assessment":
            self._generate_quality_assessment_report()
        elif report_type == "Custom Report":
            self._generate_custom_report()
    
    def _calculate_spectrum_statistics(self, spectrum: np.ndarray, spectrum_data: Dict) -> Dict:
        """Calculate comprehensive spectrum statistics."""
        stats = {
            'Data Points': len(spectrum),
            'Mean Intensity': f"{np.mean(spectrum):.3f}",
            'Std Dev': f"{np.std(spectrum):.3f}",
            'Min Intensity': f"{np.min(spectrum):.3f}",
            'Max Intensity': f"{np.max(spectrum):.3f}",
            'Dynamic Range': f"{np.max(spectrum) - np.min(spectrum):.3f}",
            'Peak Count': len(spectrum_data.get('peak_positions', [])),
            'SNR Estimate': f"{np.mean(spectrum) / np.std(spectrum):.2f}",
        }
        return stats
    
    def _display_spectrum_metadata(self, spectrum_data: Dict):
        """Display spectrum metadata in organized format."""
        st.subheader("Spectrum Metadata")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Chemical Information:**")
            st.write(f"Compound: {spectrum_data['compound_name']}")
            st.write(f"Formula: {spectrum_data['chemical_formula'] or 'N/A'}")
            st.write(f"CAS Number: {spectrum_data['cas_number'] or 'N/A'}")
        
        with col2:
            st.write("**Measurement Details:**")
            st.write(f"Laser: {spectrum_data['laser_wavelength']} nm")
            st.write(f"Integration: {spectrum_data['integration_time']} ms")
            st.write(f"Date: {spectrum_data['acquisition_date']}")
    
    def _display_comparison_statistics(self, spectra: List[np.ndarray], spectra_info: List[Dict]):
        """Display comparison statistics for multiple spectra."""
        st.subheader("Comparison Statistics")
        
        if len(spectra) < 2:
            return
        
        # Calculate cross-correlation matrix
        correlation_matrix = np.corrcoef(spectra)
        
        # Create correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            labels=dict(x="Spectrum", y="Spectrum", color="Correlation"),
            x=[spec['compound_name'] for spec in spectra_info],
            y=[spec['compound_name'] for spec in spectra_info],
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        fig_corr.update_layout(title="Spectral Correlation Matrix", height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Min Correlation", f"{np.min(correlation_matrix[correlation_matrix < 1]):.3f}")
        
        with col2:
            st.metric("Mean Correlation", f"{np.mean(correlation_matrix[correlation_matrix < 1]):.3f}")
        
        with col3:
            st.metric("Max Correlation", f"{np.max(correlation_matrix[correlation_matrix < 1]):.3f}")
    
    def _calculate_variation_statistics(self, sample_spectra: List[Dict], compound_name: str):
        """Calculate variation statistics for multiple measurements."""
        spectra_array = np.array([spec['data'] for spec in sample_spectra])
        
        mean_spectrum = np.mean(spectra_array, axis=0)
        std_spectrum = np.std(spectra_array, axis=0)
        
        # Overall variation metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Variation", f"{np.mean(std_spectrum):.4f}")
        
        with col2:
            st.metric("Max Variation", f"{np.max(std_spectrum):.4f}")
        
        with col3:
            coefficient_of_variation = np.mean(std_spectrum) / np.mean(mean_spectrum)
            st.metric("Coeff. of Variation", f"{coefficient_of_variation:.4f}")
    
    def _generate_database_summary_report(self):
        """Generate comprehensive database summary report."""
        st.write("Generating database summary report...")
        # Implementation would create PDF or detailed analysis
        st.success("Report generation feature coming soon!")
    
    def _generate_spectrum_analysis_report(self):
        """Generate detailed spectrum analysis report."""
        st.write("Generating spectrum analysis report...")
        st.success("Report generation feature coming soon!")
    
    def _generate_quality_assessment_report(self):
        """Generate data quality assessment report."""
        st.write("Generating quality assessment report...")
        st.success("Report generation feature coming soon!")
    
    def _generate_custom_report(self):
        """Generate custom user-defined report."""
        st.write("Custom report builder coming soon!")
        st.success("Report generation feature coming soon!")