import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
from src.aes_utils import AESHelper

class CPAAnalyzer:
    """Correlation Power Analysis implementation for AES key recovery."""
    
    def __init__(self, leakage_model: str = "hamming_weight"):
        """
        Initialize CPA analyzer.
        
        Args:
            leakage_model: Type of leakage model ("hamming_weight", "hamming_distance", "identity")
        """
        self.leakage_model = leakage_model
        self.aes_helper = AESHelper()
        
    def perform_cpa(self, traces: np.ndarray, plaintexts: Optional[np.ndarray], 
                   key_byte_position: int) -> Dict:
        """
        Perform Correlation Power Analysis to recover a key byte.
        
        Args:
            traces: Power traces array of shape (num_traces, num_samples)
            plaintexts: Plaintext inputs of shape (num_traces, 16)
            key_byte_position: Position of the target key byte (0-15)
            
        Returns:
            Dictionary containing analysis results
        """
        if plaintexts is None:
            # Generate random plaintexts if none provided
            plaintexts = np.random.randint(0, 256, size=(len(traces), 16), dtype=np.uint8)
        
        num_traces, num_samples = traces.shape
        correlations = np.zeros((256, num_samples))
        
        # Test all possible key byte values
        for key_guess in range(256):
            # Calculate hypothetical intermediate values
            intermediate_values = self._calculate_intermediate_values(
                plaintexts[:, key_byte_position], key_guess
            )
            
            # Apply leakage model
            leakage_hypothesis = self._apply_leakage_model(intermediate_values)
            
            # Calculate correlation for each sample point
            for sample_idx in range(num_samples):
                correlation = self._pearson_correlation(
                    leakage_hypothesis, traces[:, sample_idx]
                )
                correlations[key_guess, sample_idx] = abs(correlation)
        
        # Find best key candidate
        max_correlations = np.max(correlations, axis=1)
        best_key = np.argmax(max_correlations)
        max_correlation = max_correlations[best_key]
        peak_position = np.argmax(correlations[best_key])
        
        # Calculate confidence
        sorted_correlations = np.sort(max_correlations)
        confidence = ((sorted_correlations[-1] - sorted_correlations[-2]) / 
                     sorted_correlations[-1] * 100)
        
        return {
            'correlations': correlations,
            'best_key': best_key,
            'max_correlation': max_correlation,
            'peak_position': peak_position,
            'confidence': confidence,
            'key_byte_position': key_byte_position,
            'correlation_traces': correlations[best_key]
        }
    
    def perform_full_key_recovery(self, traces: np.ndarray, plaintexts: Optional[np.ndarray]) -> Dict:
        """
        Perform CPA on all 16 key bytes.
        
        Returns:
            Dictionary with results for each key byte
        """
        results = {}
        recovered_key = np.zeros(16, dtype=np.uint8)
        total_confidence = 0
        
        progress_placeholder = st.empty()
        
        for byte_pos in range(16):
            with progress_placeholder.container():
                st.write(f"Analyzing key byte {byte_pos + 1}/16...")
            
            byte_result = self.perform_cpa(traces, plaintexts, byte_pos)
            results[f'byte_{byte_pos}'] = byte_result
            recovered_key[byte_pos] = byte_result['best_key']
            total_confidence += byte_result['confidence']
        
        progress_placeholder.empty()
        
        results['full_key'] = recovered_key
        results['average_confidence'] = total_confidence / 16
        
        return results
    
    def _calculate_intermediate_values(self, plaintext_bytes: np.ndarray, key_guess: int) -> np.ndarray:
        """Calculate intermediate values (typically S-box output)."""
        # AES S-box operation: S-box[plaintext XOR key]
        xor_values = plaintext_bytes ^ key_guess
        return self.aes_helper.sbox_lookup(xor_values)
    
    def _apply_leakage_model(self, intermediate_values: np.ndarray) -> np.ndarray:
        """Apply leakage model to intermediate values."""
        if self.leakage_model == "hamming_weight":
            return self._hamming_weight(intermediate_values)
        elif self.leakage_model == "hamming_distance":
            # For simplicity, use HD from zero
            return self._hamming_weight(intermediate_values)
        elif self.leakage_model == "identity":
            return intermediate_values.astype(np.float32)
        else:
            return self._hamming_weight(intermediate_values)
    
    def _hamming_weight(self, values: np.ndarray) -> np.ndarray:
        """Calculate Hamming weight of values."""
        return np.array([bin(v).count('1') for v in values], dtype=np.float32)
    
    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y):
            return 0.0
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def analyze_correlation_quality(self, correlations: np.ndarray) -> Dict:
        """Analyze the quality of correlation traces."""
        # Find peaks and their properties
        max_correlations = np.max(correlations, axis=1)
        best_keys = np.argsort(max_correlations)[-5:]  # Top 5 candidates
        
        # Signal-to-noise ratio of correlation
        signal_power = np.var(max_correlations)
        noise_power = np.mean([np.var(correlations[key]) for key in range(256)])
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        # Peak sharpness
        best_key = np.argmax(max_correlations)
        peak_trace = correlations[best_key]
        peak_pos = np.argmax(peak_trace)
        
        # Find full width at half maximum (FWHM)
        half_max = peak_trace[peak_pos] / 2
        indices = np.where(peak_trace >= half_max)[0]
        fwhm = len(indices) if len(indices) > 0 else 1
        
        return {
            'snr': snr,
            'top_candidates': best_keys,
            'peak_sharpness': peak_trace[peak_pos] / np.mean(peak_trace),
            'fwhm': fwhm,
            'distinguishability': max_correlations[best_key] / np.mean(max_correlations)
        }
    
    def differential_analysis(self, traces: np.ndarray, plaintexts: np.ndarray, 
                            key_byte_position: int) -> Dict:
        """
        Perform differential analysis to identify points of interest.
        
        Args:
            traces: Power traces
            plaintexts: Corresponding plaintexts
            key_byte_position: Target key byte position
            
        Returns:
            Dictionary with differential analysis results
        """
        # Group traces by Hamming weight of target byte
        target_bytes = plaintexts[:, key_byte_position]
        hw_groups = {}
        
        for hw in range(9):  # Hamming weights 0-8
            mask = np.array([bin(b).count('1') == hw for b in target_bytes])
            if np.sum(mask) > 0:
                hw_groups[hw] = traces[mask]
        
        # Calculate mean traces for each group
        mean_traces = {}
        for hw, group_traces in hw_groups.items():
            mean_traces[hw] = np.mean(group_traces, axis=0)
        
        # Calculate variance between groups
        all_means = np.array(list(mean_traces.values()))
        variance_trace = np.var(all_means, axis=0)
        
        # Find points of interest (POI)
        threshold = np.percentile(variance_trace, 95)  # Top 5% variance points
        poi_indices = np.where(variance_trace >= threshold)[0]
        
        return {
            'variance_trace': variance_trace,
            'poi_indices': poi_indices,
            'mean_traces': mean_traces,
            'hw_groups_sizes': {hw: len(traces) for hw, traces in hw_groups.items()}
        }
