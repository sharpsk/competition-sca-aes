import numpy as np
import pandas as pd
import io
from typing import List, Tuple, Optional, Union
import streamlit as st

class TraceProcessor:
    """Handles loading and preprocessing of power trace data."""
    
    def __init__(self):
        self.supported_formats = ['npy', 'csv', 'txt', 'dat']
    
    def load_traces(self, uploaded_files, format_type: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load traces from uploaded files.
        
        Returns:
            Tuple of (traces, plaintexts, labels) where each can be None if not available
        """
        try:
            traces = []
            plaintexts = []
            labels = []
            
            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if format_type == "Auto-detect":
                    format_to_use = self._detect_format(uploaded_file, file_extension)
                else:
                    format_to_use = format_type
                
                trace_data, plaintext_data, label_data = self._load_single_file(
                    uploaded_file, format_to_use
                )
                
                if trace_data is not None:
                    traces.append(trace_data)
                    if plaintext_data is not None:
                        plaintexts.append(plaintext_data)
                    if label_data is not None:
                        labels.append(label_data)
            
            if traces:
                # Concatenate all traces
                traces_array = np.vstack(traces) if len(traces) > 1 else traces[0]
                plaintexts_array = np.vstack(plaintexts) if plaintexts and len(plaintexts) > 1 else (plaintexts[0] if plaintexts else None)
                labels_array = np.concatenate(labels) if labels and len(labels) > 1 else (labels[0] if labels else None)
                
                return traces_array, plaintexts_array, labels_array
            
            return None, None, None
            
        except Exception as e:
            st.error(f"Error loading traces: {str(e)}")
            return None, None, None
    
    def _detect_format(self, uploaded_file, file_extension: str) -> str:
        """Auto-detect the file format based on extension and content."""
        if file_extension == 'npy':
            return "NumPy array (.npy)"
        elif file_extension == 'csv':
            return "CSV with headers"
        elif file_extension in ['txt', 'dat']:
            # Try to determine if it's structured text or binary
            try:
                content = uploaded_file.read(1024)
                uploaded_file.seek(0)  # Reset file pointer
                
                # Check if content is mostly printable ASCII
                if all(32 <= byte <= 126 or byte in [9, 10, 13] for byte in content):
                    return "Raw text"
                else:
                    return "Binary (.dat)"
            except:
                return "Raw text"
        
        return "Raw text"
    
    def _load_single_file(self, uploaded_file, format_type: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single file based on its format."""
        try:
            if format_type == "NumPy array (.npy)":
                return self._load_numpy_file(uploaded_file)
            elif format_type == "CSV with headers":
                return self._load_csv_file(uploaded_file)
            elif format_type == "Raw text":
                return self._load_text_file(uploaded_file)
            elif format_type == "Binary (.dat)":
                return self._load_binary_file(uploaded_file)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
            return None, None, None
    
    def _load_numpy_file(self, uploaded_file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load NumPy file."""
        data = np.load(uploaded_file, allow_pickle=True)
        
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                # Single trace
                traces = data.reshape(1, -1)
            else:
                traces = data
            return traces, None, None
        elif isinstance(data, dict) or hasattr(data, 'files'):
            # NPZ file or structured array
            traces = data.get('traces', data.get('power_traces', None))
            plaintexts = data.get('plaintexts', data.get('inputs', None))
            labels = data.get('labels', data.get('keys', None))
            
            return traces, plaintexts, labels
        
        return None, None, None
    
    def _load_csv_file(self, uploaded_file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load CSV file."""
        df = pd.read_csv(uploaded_file)
        
        # Try to identify columns
        trace_cols = [col for col in df.columns if 'trace' in col.lower() or 'power' in col.lower() or col.startswith('sample')]
        plaintext_cols = [col for col in df.columns if 'plain' in col.lower() or 'input' in col.lower()]
        label_cols = [col for col in df.columns if 'key' in col.lower() or 'label' in col.lower()]
        
        traces = None
        plaintexts = None
        labels = None
        
        if trace_cols:
            traces = df[trace_cols].values
        elif len(df.columns) > 16:  # Assume most columns are trace samples
            # Take all numeric columns except the first 16 (likely plaintexts/keys)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 16:
                traces = df[numeric_cols[16:]].values
                if len(numeric_cols) >= 16:
                    plaintexts = df[numeric_cols[:16]].values
        
        if plaintext_cols:
            plaintexts = df[plaintext_cols].values
        
        if label_cols:
            labels = df[label_cols].values.flatten() if len(label_cols) == 1 else df[label_cols].values
        
        return traces, plaintexts, labels
    
    def _load_text_file(self, uploaded_file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load plain text file."""
        content = uploaded_file.read().decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Try to parse as space/comma separated values
        try:
            traces = []
            for line in lines:
                # Split by comma or space
                if ',' in line:
                    values = [float(x.strip()) for x in line.split(',')]
                else:
                    values = [float(x) for x in line.split()]
                traces.append(values)
            
            traces_array = np.array(traces)
            return traces_array, None, None
            
        except ValueError:
            # If parsing fails, try hex format
            try:
                traces = []
                for line in lines:
                    # Assume hex values
                    hex_values = line.replace('0x', '').split()
                    values = [int(x, 16) for x in hex_values]
                    traces.append(values)
                
                traces_array = np.array(traces, dtype=np.float32)
                return traces_array, None, None
                
            except ValueError:
                return None, None, None
    
    def _load_binary_file(self, uploaded_file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load binary file."""
        content = uploaded_file.read()
        
        # Try different data types
        for dtype in [np.float32, np.float64, np.int16, np.int32, np.uint8, np.uint16]:
            try:
                data = np.frombuffer(content, dtype=dtype)
                if len(data) > 0:
                    # Try to reshape into reasonable trace lengths
                    possible_lengths = [1000, 2000, 5000, 10000, len(data)]
                    for length in possible_lengths:
                        if len(data) % length == 0:
                            traces = data.reshape(-1, length)
                            return traces.astype(np.float32), None, None
                    
                    # If no clean division, just return as single trace
                    traces = data.reshape(1, -1).astype(np.float32)
                    return traces, None, None
                    
            except Exception:
                continue
        
        return None, None, None
    
    def preprocess_traces(self, traces: np.ndarray, method: str = "normalize") -> np.ndarray:
        """
        Preprocess traces using various methods.
        
        Args:
            traces: Input traces array
            method: Preprocessing method ("normalize", "standardize", "mean_center")
        """
        if method == "normalize":
            # Normalize each trace to [0, 1]
            traces_min = np.min(traces, axis=1, keepdims=True)
            traces_max = np.max(traces, axis=1, keepdims=True)
            return (traces - traces_min) / (traces_max - traces_min + 1e-8)
        
        elif method == "standardize":
            # Z-score normalization
            traces_mean = np.mean(traces, axis=1, keepdims=True)
            traces_std = np.std(traces, axis=1, keepdims=True)
            return (traces - traces_mean) / (traces_std + 1e-8)
        
        elif method == "mean_center":
            # Mean centering
            traces_mean = np.mean(traces, axis=1, keepdims=True)
            return traces - traces_mean
        
        return traces
    
    def generate_synthetic_plaintexts(self, num_traces: int) -> np.ndarray:
        """Generate synthetic random plaintexts if none provided."""
        return np.random.randint(0, 256, size=(num_traces, 16), dtype=np.uint8)
