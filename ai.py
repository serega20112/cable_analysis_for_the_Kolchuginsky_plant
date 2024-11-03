import cv2
import numpy as np
import os
import json
from datetime import datetime
import requests
from PIL import Image, ImageTk
from io import BytesIO
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple, Union
from bs4 import BeautifulSoup
import re
import urllib.parse
import urllib.request
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import base64
import http.client
from collections import defaultdict
import colorsys

class WireFeatureExtractor:
    """Advanced feature extraction for wire cross-sections."""
    
    def __init__(self):
        self.feature_weights = {
            'texture': 0.3,
            'shape': 0.25,
            'color': 0.25,
            'gradient': 0.2
        }
        
        self.texture_params = {
            'kernel_sizes': [(3,3), (5,5), (7,7)],
            'orientations': 8,
            'scales': 3,
            'hist_bins': 10
        }
        
        self.shape_params = {
            'num_points': 32,
            'min_area': 100,
            'max_area': 10000
        }
        
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract multi-scale texture features using Gabor filters."""
        features = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply multiple Gabor filters
        for theta in np.arange(0, np.pi, np.pi / self.texture_params['orientations']):
            for sigma in range(1, self.texture_params['scales'] + 1):
                kernel = cv2.getGaborKernel(
                    (21, 21), sigma=sigma, theta=theta,
                    lambd=10, gamma=0.5, psi=0
                )
                filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
                features.append(np.mean(filtered))
                features.append(np.std(filtered))
        
        # Add LBP features
        for kernel_size in self.texture_params['kernel_sizes']:
            lbp = self._compute_lbp(gray, kernel_size)
            hist, _ = np.histogram(lbp.ravel(), bins=self.texture_params['hist_bins'])
            features.extend(hist.tolist())
        
        return np.array(features)
    
    def _compute_lbp(self, image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
        """Compute Local Binary Pattern."""
        rows, cols = image.shape
        result = np.zeros_like(image)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                binary = (image[i-1:i+2, j-1:j+2] >= center).astype(int)
                binary[1,1] = 0  # exclude center
                result[i, j] = np.sum(binary * (2 ** np.arange(8)))
                
        return result
    
    def extract_shape_features(self, contour: np.ndarray) -> np.ndarray:
        """Extract comprehensive shape features."""
        features = []
        
        # Basic shape metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features.append(4 * np.pi * area / (perimeter * perimeter))  # circularity
        
        # Fit ellipse and get parameters
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            features.extend([
                ellipse[1][0] / ellipse[1][1],  # aspect ratio
                ellipse[2]  # orientation
            ])
        else:
            features.extend([1.0, 0.0])
        
        # Convex hull features
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        features.append(area / hull_area)  # solidity
        
        # Moment invariants
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments)
        
        # Fourier descriptors
        fd = self._compute_fourier_descriptors(contour)
        features.extend(fd)
        
        return np.array(features)
    
    def _compute_fourier_descriptors(self, contour: np.ndarray) -> np.ndarray:
        """Compute normalized Fourier descriptors."""
        contour = contour.reshape(-1, 2)
        contour_complex = contour[:,0] + 1j * contour[:,1]
        
        # Compute FFT and normalize
        fft_result = np.fft.fft(contour_complex)
        descriptors = np.abs(fft_result)[1:self.shape_params['num_points']+1]
        
        # Scale invariance
        if len(descriptors) > 0:
            descriptors = descriptors / descriptors[0]
        
        return descriptors
    
    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color features using multiple color spaces."""
        features = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Compute color histograms
        for color_space in [image, hsv, lab]:
            for channel in range(3):
                hist = cv2.calcHist([color_space], [channel], None, [8], [0, 256])
                features.extend(hist.flatten())
        
        # Add color moments
        for color_space in [image, hsv, lab]:
            for channel in range(3):
                channel_data = color_space[:,:,channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.cbrt(np.mean(np.power(channel_data - np.mean(channel_data), 3)))
                ])
        
        # Color coherence vector
        ccv = self._compute_color_coherence(image)
        features.extend(ccv)
        
        return np.array(features)
    
    def _compute_color_coherence(self, image: np.ndarray, threshold: int = 30) -> np.ndarray:
        """Compute color coherence vector."""
        coherent = np.zeros(8, dtype=np.float32)
        incoherent = np.zeros(8, dtype=np.float32)
        
        # Quantize colors
        quantized = image // 32
        
        # Find connected components
        for color in range(8):
            mask = (quantized[:,:,0] == color).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            
            for i in range(1, num_labels):
                size = stats[i, cv2.CC_STAT_AREA]
                if size > threshold:
                    coherent[color] += size
                else:
                    incoherent[color] += size
        
        # Normalize
        total = np.sum(coherent) + np.sum(incoherent)
        if total > 0:
            coherent = coherent / total
            incoherent = incoherent / total
        
        return np.concatenate([coherent, incoherent])
    
    def extract_gradient_features(self, image: np.ndarray) -> np.ndarray:
        """Extract gradient-based features."""
        features = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        mag = np.sqrt(gx*gx + gy*gy)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        
        # Histogram of oriented gradients
        hist_mag, _ = np.histogram(mag, bins=9)
        hist_angle, _ = np.histogram(angle, bins=9, range=(-180, 180))
        
        features.extend(hist_mag)
        features.extend(hist_angle)
        
        # Add statistical measures
        for grad in [mag, angle]:
            features.extend([
                np.mean(grad),
                np.std(grad),
                np.percentile(grad, 25),
                np.percentile(grad, 75)
            ])
        
        return np.array(features)

class WireImageProcessor:
    """Advanced image processing for wire detection."""
    
    def __init__(self):
        self.preprocessing_params = {
            'blur_kernel': (5, 5),
            'clahe_clip': 2.0,
            'clahe_grid': (8, 8),
            'morph_kernel': np.ones((3, 3), np.uint8),
            'canny_low': 50,
            'canny_high': 150
        }
        
        self.segmentation_params = {
            'block_size': 11,
            'c': 2,
            'min_area': 100,
            'max_area': 10000
        }
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced preprocessing techniques."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.preprocessing_params['clahe_clip'],
            tileGridSize=self.preprocessing_params['clahe_grid']
        )
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Apply bilateral filter for edge preservation
        bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(
            bilateral,
            self.preprocessing_params['blur_kernel'],
            0
        )
        
        return blurred
    
    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Perform advanced image segmentation."""
        preprocessed = self.preprocess_image(image)
        
        # Multi-scale segmentation
        masks = []
        contours = []
        
        # Apply different thresholding techniques
        binary_masks = self._apply_thresholding(preprocessed)
        
        # Combine masks
        combined_mask = np.zeros_like(preprocessed)
        for mask in binary_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations
        cleaned = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_CLOSE,
            self.preprocessing_params['morph_kernel']
        )
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            self.preprocessing_params['morph_kernel']
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        
        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.segmentation_params['min_area'] < area < self.segmentation_params['max_area']:
                filtered_contours.append(contour)
        
        return cleaned, filtered_contours
    
    def _apply_thresholding(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply multiple thresholding techniques."""
        masks = []
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.segmentation_params['block_size'],
            self.segmentation_params['c']
        )
        masks.append(adaptive)
        
        # Otsu's thresholding
        _, otsu = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        masks.append(otsu)
        
        # Multi-level thresholding
        for i in range(3):
            thresh_value = 85 + i * 85
            _, binary = cv2.threshold(
                image,
                thresh_value,
                255,
                cv2.THRESH_BINARY_INV
            )
            masks.append(binary)
        
        return masks

class WireAnalyzer:
    """Main class for wire cross-section analysis."""
    
    def __init__(self):
        self.feature_extractor = WireFeatureExtractor()
        self.image_processor = WireImageProcessor()
        self.reference_features = None
        self.reference_image = None
        self.similarity_threshold = 0.85
        
        # Initialize paths
        self.data_dir = "wire_data"
        self.cache_dir = os.path.join(self.data_dir, "cache")
        self.results_dir = os.path.join(self.data_dir, "results")
        
        # Create directories
        for directory in [self.data_dir, self.cache_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('wire_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def set_reference(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """Set reference region for comparison."""
        try:
            x, y, w, h = roi
            reference_region = image[y:y+h, x:x+w]
            
            # Extract features from reference
            self.reference_image = reference_region.copy()
            self.reference_features = self._extract_all_features(reference_region)
            
            # Save reference data
            cv2.imwrite(os.path.join(self.data_dir, "reference.jpg"), reference_region)
            np.save(os.path.join(self.data_dir, "reference_features.npy"), self.reference_features)
            
            return True
        except Exception as e:
            self.logger.error(f"Error setting reference: {e}")
            return False
    
    def _extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """Extract all features from an image region."""
        # Segment the image
        mask, contours = self.image_processor.segment_image(image)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract features
        texture_features = self.feature_extractor.extract_texture_features(image)
        shape_features = self.feature_extractor.extract_shape_features(largest_contour)
        color_features = self.feature_extractor.extract_color_features(image)
        gradient_features = self.feature_extractor.extract_gradient_features(image)
        
        # Combine features
        return np.concatenate([
            texture_features,
            shape_features,
            color_features,
            gradient_features
        ])
    
    def find_similar_regions(self, image: np.ndarray) -> List[Dict]:
        """Find regions similar to reference in the image."""
        if self.reference_features is None:
            self.logger.error("No reference region set")
            return []
        
        similar_regions = []
        height, width = image.shape[:2]
        ref_h, ref_w = self.reference_image.shape[:2]
        
        # Multi-scale sliding window
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        for scale in scales:
            scaled_ref_w = int(ref_w * scale)
            scaled_ref_h = int(ref_h * scale)
            
            if scaled_ref_w > width or scaled_ref_h > height:
                continue
            
            step_size = min(scaled_ref_w, scaled_ref_h) // 2
            
            for y in range(0, height - scaled_ref_h + 1, step_size):
                for x in range(0, width - scaled_ref_w + 1, step_size):
                    roi = image[y:y+scaled_ref_h, x:x+scaled_ref_w]
                    features = self._extract_all_features(roi)
                    
                    if features is not None:
                        similarity = self._calculate_similarity(features)
                        
                        if similarity >= self.similarity_threshold:
                            similar_regions.append({
                                'region': (x, y, scaled_ref_w, scaled_ref_h),
                                'similarity': similarity,
                                'scale': scale
                            })
        
        # Non-maximum suppression
        return self._non_max_suppression(similar_regions)
    
    def _calculate_similarity(self, features: np.ndarray) -> float:
        """Calculate similarity between feature sets."""
        if features is None or self.reference_features is None:
            return 0.0
        
        # Normalize features
        features_norm = features / (np.linalg.norm(features) + 1e-7)
        ref_features_norm = self.reference_features / (np.linalg.norm(self.reference_features) + 1e-7)
        
        # Calculate cosine similarity
        similarity = np.dot(features_norm, ref_features_norm)
        
        return max(0.0, min(1.0, similarity))
    
    def _non_max_suppression(self, regions: List[Dict], overlap_thresh: float = 0.3) -> List[Dict]:
        """Apply non-maximum suppression to overlapping regions."""
        if not regions:
            return []
        
        # Convert regions to numpy array
        boxes = np.array([list(r['region']) for r in regions])
        scores = np.array([r['similarity'] for r in regions])
        
        # Compute areas
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,0] + boxes[:,2]
        y2 = boxes[:,1] + boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by score
        idxs = np.argsort(scores)[::-1]
        
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            
            # Find overlapping boxes
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / areas[idxs[1:]]
            
            # Remove overlapping boxes
            idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
        
        return [regions[i] for i in keep]
    
    def download_similar_images(self, query: str, max_images: int = 20) -> List[str]:
        """Download similar images from the internet."""
        try:
            # Prepare search query
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Make request
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                html = response.read()
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            img_tags = soup.find_all('img')
            
            # Extract image URLs
            urls = []
            for img in img_tags:
                img_url = img.get('src')
                if img_url and img_url.startswith('http'):
                    urls.append(img_url)
            
            # Download images
            downloaded = []
            for i, img_url in enumerate(urls[:max_images]):
                try:
                    response = requests.get(img_url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        img_path = os.path.join(self.data_dir, f"downloaded_{i}.jpg")
                        with open(img_path, 'wb') as f:
                            f.write(response.content)
                        downloaded.append(img_path)
                        time.sleep(0.5)  # Respect rate limits
                except Exception as e:
                    self.logger.error(f"Error downloading {img_url}: {e}")
                    continue
            
            return downloaded
            
        except Exception as e:
            self.logger.error(f"Error downloading images: {e}")
            return []
    
    def analyze_directory(self, directory: str) -> List[Dict]:
        """Analyze all images in a directory."""
        results = []
        image_files = [
            f for f in os.listdir(directory)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        for filename in image_files:
            try:
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                regions = self.find_similar_regions(image)
                
                if regions:
                    # Draw results
                    result_image = image.copy()
                    for region in regions:
                        x, y, w, h = region['region']
                        similarity = region['similarity']
                        color = self._get_similarity_color(similarity)
                        cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(
                            result_image,
                            f"{similarity:.2f}",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
                    
                    # Save result
                    result_path = os.path.join(
                        self.results_dir,
                        f"result_{os.path.basename(filename)}"
                    )
                    cv2.imwrite(result_path, result_image)
                    
                    results.append({
                        'image_path': image_path,
                        'result_path': result_path,
                        'regions': regions
                    })
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {filename}: {e}")
                continue
        
        return results
    
    def _get_similarity_color(self, similarity: float) -> Tuple[int, int, int]:
        """Get color based on similarity score."""
        hue = similarity * 120  # Red to Green
        rgb = colorsys.hsv_to_rgb(hue/360, 1, 1)
        return tuple(int(x * 255) for x in rgb)

class WireAnalyzerGUI:
    """Graphical user interface for wire analyzer."""
    
    def __init__(self):
        self.analyzer = WireAnalyzer()
        self.current_image = None
        self.current_result = None
        self.roi_selecting = False
        self.roi_start = None
        self.current_roi = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the GUI window and widgets."""
        self.root = tk.Tk()
        self.root.title("Wire Cross-Section Analyzer")
        self.root.geometry("1200x800")
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create control panel
        self.control_panel = ttk.Frame(self.main_container)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Create buttons
        self.create_buttons()
        
        # Create progress bar
        self.progress = ttk.Progressbar(
            self.control_panel,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(pady=5)
        
        # Create results text area
        self.results_text = tk.Text(
            self.control_panel,
            height=20,
            width=40,
            wrap=tk.WORD
        )
        self.results_text.pack(pady=5)
        
        # Add scrollbar to results
        scrollbar = ttk.Scrollbar(
            self.control_panel,
            orient="vertical",
            command=self.results_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Create image canvas
        self.canvas = tk.Canvas(
            self.main_container,
            bg='white',
            width=800,
            height=600
        )
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Bind events
        self.bind_events()
        
        self.root.mainloop()
    
    def create_buttons(self):
        """Create control buttons."""
        buttons = [
            ("Load Image", self.load_image),
            ("Set Reference", self.start_roi_selection),
            ("Process Directory", self.select_directory),
            ("Download Similar", self.download_similar),
            ("Clear Results", self.clear_results),
            ("Save Analysis", self.save_analysis)
        ]
        
        for text, command in buttons:
            ttk.Button(
                self.control_panel,
                text=text,
                command=command
            ).pack(pady=5, fill=tk.X)
    
    def bind_events(self):
        """Bind mouse and keyboard events."""
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Configure>', self.on_canvas_resize)
    
    def load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                self.current_result = None
                self.display_image()
                self.log_message(f"Loaded image: {file_path}")
            except Exception as e:
                self.log_message(f"Error loading image: {e}", error=True)
    
    def start_roi_selection(self):
        """Start ROI selection mode."""
        if self.current_image is None:
            self.log_message("Please load an image first", error=True)
            return
        
        self.roi_selecting = True
        self.current_roi = None
        self.log_message("Click and drag to select reference region")
    
    def select_directory(self):
        """Select directory for batch processing."""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.progress.start()
            threading.Thread(
                target=self.process_directory_thread,
                args=(dir_path,)
            ).start()
    
    def download_similar(self):
        """Download similar images from the internet."""
        if not self.current_image:
            self.log_message("Please load an image first", error=True)
            return
        
        query = "wire cross section microscope"
        self.progress.start()
        threading.Thread(
            target=self.download_similar_thread,
            args=(query,)
        ).start()
    
    def process_directory_thread(self, dir_path: str):
        """Thread for processing directory."""
        try:
            results = self.analyzer.analyze_directory(dir_path)
            self.root.after(0, self.on_processing_complete, results)
        except Exception as e:
            self.root.after(0, self.log_message, f"Error processing directory: {e}", True)
        finally:
            self.root.after(0, self.progress.stop)
    
    def download_similar_thread(self, query: str):
        """Thread for downloading similar images."""
        try:
            downloaded = self.analyzer.download_similar_images(query)
            self.root.after(0, self.on_download_complete, downloaded)
        except Exception as e:
            self.root.after(0, self.log_message, f"Error downloading images: {e}", True)
        finally:
            self.root.after(0, self.progress.stop)
    
    def on_processing_complete(self, results: List[Dict]):
        """Handle completion of directory processing."""
        self.log_message(f"Processing complete. Found {len(results)} images with similar regions")
        for result in results:
            self.log_message(f"Results saved to {result['result_path']}")
    
    def on_download_complete(self, downloaded: List[str]):
        """Handle completion of image```python
    def on_download_complete(self, downloaded: List[str]):
        """Handle completion of image downloads."""
        self.log_message(f"Downloaded {len(downloaded)} similar images")
        for path in downloaded:
            self.log_message(f"Saved to {path}")
    
    def clear_results(self):
        """Clear results text area."""
        self.results_text.delete(1.0, tk.END)
    
    def save_analysis(self):
        """Save current analysis results."""
        if not self.current_result:
            self.log_message("No analysis results to save", error=True)
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(self.current_result, f, indent=2)
                self.log_message(f"Analysis saved to {file_path}")
        except Exception as e:
            self.log_message(f"Error saving analysis: {e}", error=True)
    
    def display_image(self):
        """Display current image on canvas."""
        if self.current_image is None:
            return
        
        try:
            display_image = cv2.cvtColor(
                self.current_image if self.current_result is None else self.current_result,
                cv2.COLOR_BGR2RGB
            )
            
            # Resize to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            img_height, img_width = display_image.shape[:2]
            scale = min(
                canvas_width/img_width,
                canvas_height/img_height
            )
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            display_image = cv2.resize(
                display_image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
            
            self.photo = ImageTk.PhotoImage(
                Image.fromarray(display_image)
            )
            
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width//2,
                canvas_height//2,
                anchor=tk.CENTER,
                image=self.photo
            )
            
            if self.current_roi:
                self.draw_roi()
                
        except Exception as e:
            self.log_message(f"Error displaying image: {e}", error=True)
    
    def draw_roi(self):
        """Draw current ROI on canvas."""
        if not self.current_roi:
            return
            
        x, y, w, h = self.current_roi
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Scale ROI coordinates
        scale = min(
            canvas_width/self.current_image.shape[1],
            canvas_height/self.current_image.shape[0]
        )
        
        x = int(x * scale)
        y = int(y * scale)
        w = int(w * scale)
        h = int(h * scale)
        
        self.canvas.create_rectangle(
            x, y, x+w, y+h,
            outline="red",
            width=2,
            tags="roi"
        )
    
    def on_mouse_down(self, event):
        """Handle mouse button press."""
        if self.roi_selecting:
            self.roi_start = (event.x, event.y)
    
    def on_mouse_move(self, event):
        """Handle mouse movement."""
        if self.roi_selecting and self.roi_start:
            self.canvas.delete("roi")
            self.canvas.create_rectangle(
                self.roi_start[0], self.roi_start[1],
                event.x, event.y,
                outline="red",
                tags="roi"
            )
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        if self.roi_selecting and self.roi_start:
            x1, y1 = self.roi_start
            x2, y2 = event.x, event.y
            
            # Convert to image coordinates
            scale = self.current_image.shape[1] / self.photo.width()
            
            self.current_roi = (
                int(min(x1, x2) * scale),
                int(min(y1, y2) * scale),
                int(abs(x2 - x1) * scale),
                int(abs(y2 - y1) * scale)
            )
            
            self.roi_selecting = False
            
            if self.analyzer.set_reference(self.current_image, self.current_roi):
                self.log_message("Reference region set successfully")
                self.draw_roi()
            else:
                self.log_message("Failed to set reference region", error=True)
    
    def on_canvas_resize(self, event):
        """Handle canvas resize event."""
        if self.current_image is not None:
            self.display_image()
    
    def log_message(self, message: str, error: bool = False):
        """Log message to results text area."""
        self.results_text.insert(tk.END, f"\n{message}")
        if error:
            self.results_text.tag_add(
                "error",
                "end-2c linestart",
                "end-1c"
            )
            self.results_text.tag_config(
                "error",
                foreground="red"
            )
        self.results_text.see(tk.END)

def main():
    """Main entry point."""
    app = WireAnalyzerGUI()

if __name__ == "__main__":