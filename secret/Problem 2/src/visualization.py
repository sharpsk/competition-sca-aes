import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

class TraceVisualizer:
    """Visualization utilities for side channel analysis."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set1
    
    def plot_sample_traces(self, traces: np.ndarray, max_traces: int = 10) -> go.Figure:
        """
        Plot sample power traces.
        
        Args:
            traces: Power traces array
            max_traces: Maximum number of traces to display
        """
        fig = go.Figure()
        
        num_traces_to_show = min(max_traces, len(traces))
        
        for i in range(num_traces_to_show):
            fig.add_trace(go.Scatter(
                y=traces[i],
                mode='lines',
                name=f'Trace {i+1}',
                line=dict(width=1),
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Sample Power Traces",
            xaxis_title="Sample Point",
            yaxis_title="Power Consumption",
            hovermode='x unified',
            showlegend=True if num_traces_to_show <= 5 else False
        )
        
        return fig
    
    def plot_correlation_traces(self, correlations: np.ndarray) -> go.Figure:
        """
        Plot correlation traces for all key hypotheses.
        
        Args:
            correlations: Correlation matrix (256 x num_samples)
        """
        fig = go.Figure()
        
        # Plot top 5 correlation traces
        max_correlations = np.max(correlations, axis=1)
        top_keys = np.argsort(max_correlations)[-5:]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, key in enumerate(top_keys):
            fig.add_trace(go.Scatter(
                y=correlations[key],
                mode='lines',
                name=f'Key 0x{key:02X} (max: {max_correlations[key]:.4f})',
                line=dict(color=colors[i], width=2 if i == len(top_keys)-1 else 1)
            ))
        
        fig.update_layout(
            title="Correlation Traces (Top 5 Key Candidates)",
            xaxis_title="Sample Point",
            yaxis_title="Correlation Coefficient",
            hovermode='x unified'
        )
        
        return fig
    
    def plot_correlation_heatmap(self, correlations: np.ndarray) -> go.Figure:
        """
        Plot correlation heatmap for all key hypotheses.
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlations,
            x=list(range(correlations.shape[1])),
            y=[f'0x{i:02X}' for i in range(256)],
            colorscale='Viridis',
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Heatmap (All Key Hypotheses)",
            xaxis_title="Sample Point",
            yaxis_title="Key Hypothesis",
            height=800
        )
        
        return fig
    
    def plot_power_spectrum(self, trace: np.ndarray, sampling_rate: float = 1.0) -> go.Figure:
        """
        Plot power spectral density of a trace.
        """
        # Calculate FFT
        fft = np.fft.fft(trace)
        freqs = np.fft.fftfreq(len(trace), d=1/sampling_rate)
        
        # Only plot positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        psd = np.abs(fft[:len(fft)//2])**2
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=positive_freqs,
            y=10 * np.log10(psd + 1e-10),  # Convert to dB
            mode='lines',
            name='PSD'
        ))
        
        fig.update_layout(
            title="Power Spectral Density",
            xaxis_title="Frequency",
            yaxis_title="Power (dB)",
            hovermode='x'
        )
        
        return fig
    
    def plot_feature_importance(self, importance: np.ndarray, top_n: int = 20) -> go.Figure:
        """
        Plot feature importance from ML models.
        """
        # Get top N important features
        top_indices = np.argsort(importance)[-top_n:]
        top_importance = importance[top_indices]
        
        fig = go.Figure(go.Bar(
            x=top_importance,
            y=[f'Feature {i}' for i in top_indices],
            orientation='h'
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def plot_key_ranking(self, key_scores: np.ndarray, true_key: Optional[int] = None) -> go.Figure:
        """
        Plot key ranking based on scores.
        """
        sorted_indices = np.argsort(key_scores)[::-1]
        sorted_scores = key_scores[sorted_indices]
        
        colors = ['red' if (true_key is not None and idx == true_key) else 'blue' 
                 for idx in sorted_indices]
        
        fig = go.Figure(go.Bar(
            x=[f'0x{key:02X}' for key in sorted_indices[:50]],  # Top 50
            y=sorted_scores[:50],
            marker_color=colors[:50]
        ))
        
        fig.update_layout(
            title="Key Byte Ranking (Top 50)",
            xaxis_title="Key Hypothesis",
            yaxis_title="Score",
            xaxis_tickangle=-45
        )
        
        if true_key is not None:
            fig.add_annotation(
                text="Red bar indicates true key",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)"
            )
        
        return fig
    
    def plot_trace_statistics(self, traces: np.ndarray) -> go.Figure:
        """
        Plot statistical analysis of traces.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Mean Trace", "Standard Deviation", 
                          "Min/Max Values", "Trace Distribution"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Mean trace
        mean_trace = np.mean(traces, axis=0)
        fig.add_trace(
            go.Scatter(y=mean_trace, mode='lines', name='Mean', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Standard deviation
        std_trace = np.std(traces, axis=0)
        fig.add_trace(
            go.Scatter(y=std_trace, mode='lines', name='Std Dev', line=dict(color='green')),
            row=1, col=2
        )
        
        # Min/Max traces
        min_trace = np.min(traces, axis=0)
        max_trace = np.max(traces, axis=0)
        fig.add_trace(
            go.Scatter(y=min_trace, mode='lines', name='Min', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=max_trace, mode='lines', name='Max', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Trace distribution (histogram of first sample)
        fig.add_trace(
            go.Histogram(x=traces[:, 0], name='Sample 0 Distribution', nbinsx=50),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Trace Statistical Analysis",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_differential_analysis(self, variance_trace: np.ndarray, 
                                 poi_indices: np.ndarray) -> go.Figure:
        """
        Plot differential analysis results.
        """
        fig = go.Figure()
        
        # Plot variance trace
        fig.add_trace(go.Scatter(
            y=variance_trace,
            mode='lines',
            name='Variance',
            line=dict(color='blue', width=1)
        ))
        
        # Highlight points of interest
        if len(poi_indices) > 0:
            fig.add_trace(go.Scatter(
                x=poi_indices,
                y=variance_trace[poi_indices],
                mode='markers',
                name='Points of Interest',
                marker=dict(color='red', size=6, symbol='circle')
            ))
        
        fig.update_layout(
            title="Differential Analysis - Variance Between Hamming Weight Groups",
            xaxis_title="Sample Point",
            yaxis_title="Variance",
            hovermode='x unified'
        )
        
        return fig
    
    def plot_snr_analysis(self, traces: np.ndarray, labels: np.ndarray) -> go.Figure:
        """
        Plot Signal-to-Noise Ratio analysis.
        """
        unique_labels = np.unique(labels)
        signal_variance = np.zeros(traces.shape[1])
        noise_variance = np.zeros(traces.shape[1])
        
        # Calculate signal variance (between groups)
        group_means = []
        for label in unique_labels:
            mask = labels == label
            group_mean = np.mean(traces[mask], axis=0)
            group_means.append(group_mean)
        
        group_means = np.array(group_means)
        signal_variance = np.var(group_means, axis=0)
        
        # Calculate noise variance (within groups)
        for label in unique_labels:
            mask = labels == label
            group_traces = traces[mask]
            group_mean = np.mean(group_traces, axis=0)
            group_noise = np.var(group_traces - group_mean, axis=0)
            noise_variance += group_noise / len(unique_labels)
        
        # Calculate SNR
        snr = 10 * np.log10(signal_variance / (noise_variance + 1e-10))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=snr,
            mode='lines',
            name='SNR (dB)',
            line=dict(color='green', width=2)
        ))
        
        # Add threshold line
        threshold = np.percentile(snr, 90)
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"90th percentile: {threshold:.1f} dB"
        )
        
        fig.update_layout(
            title="Signal-to-Noise Ratio Analysis",
            xaxis_title="Sample Point",
            yaxis_title="SNR (dB)",
            hovermode='x'
        )
        
        return fig
    
    def estimate_snr(self, traces: np.ndarray) -> float:
        """
        Estimate overall SNR of the traces.
        """
        mean_trace = np.mean(traces, axis=0)
        signal_power = np.var(mean_trace)
        
        noise_traces = traces - mean_trace
        noise_power = np.mean(np.var(noise_traces, axis=1))
        
        snr_linear = signal_power / (noise_power + 1e-10)
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    def create_analysis_dashboard(self, results: Dict) -> go.Figure:
        """
        Create a comprehensive analysis dashboard.
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "Correlation Traces", "Key Ranking", "Feature Importance",
                "Confidence Comparison", "Training History", "Error Analysis"
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Add plots based on available results
        # This would be customized based on the specific analysis results
        
        fig.update_layout(
            title="Side Channel Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
