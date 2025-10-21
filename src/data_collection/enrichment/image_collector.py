"""
Image collection and processing system for property listings
"""

import os
import hashlib
import requests
from PIL import Image
import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import boto3
from google.cloud import storage as gcs
from io import BytesIO
import json
import time
from loguru import logger
import re


@dataclass
class ImageMetadata:
    """Metadata for processed images"""
    url: str
    filename: str
    width: int
    height: int
    file_size: int
    format: str
    room_type: Optional[str] = None
    is_exterior: Optional[bool] = None
    quality_score: Optional[float] = None
    storage_path: Optional[str] = None
    download_error: Optional[str] = None


class ImageDownloader:
    """Downloads and processes property images"""
    
    def __init__(self, local_storage_path: str = "data/images"):
        self.local_storage_path = local_storage_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Create local storage directory
        os.makedirs(local_storage_path, exist_ok=True)
    
    def download_images(self, image_urls: List[str], listing_id: str) -> List[ImageMetadata]:
        """Download and process a list of images for a listing"""
        metadata_list = []
        
        for i, url in enumerate(image_urls):
            try:
                logger.debug(f"Downloading image {i+1}/{len(image_urls)}: {url}")
                metadata = self.download_single_image(url, listing_id, i)
                metadata_list.append(metadata)
                
                # Small delay between downloads
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error downloading image {url}: {e}")
                metadata = ImageMetadata(
                    url=url,
                    filename="",
                    width=0,
                    height=0,
                    file_size=0,
                    format="",
                    download_error=str(e)
                )
                metadata_list.append(metadata)
        
        return metadata_list
    
    def download_single_image(self, url: str, listing_id: str, index: int) -> ImageMetadata:
        """Download and process a single image"""
        try:
            # Make request with timeout
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"Invalid content type: {content_type}")
            
            # Read image data
            image_data = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                image_data.write(chunk)
            
            image_data.seek(0)
            
            # Open with PIL to validate and get metadata
            with Image.open(image_data) as img:
                width, height = img.size
                format = img.format
                
                # Generate filename
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"{listing_id}_{index:02d}_{url_hash}.{format.lower()}"
                local_path = os.path.join(self.local_storage_path, filename)
                
                # Save locally
                img.save(local_path, format=format, quality=85, optimize=True)
                
                # Get file size
                file_size = os.path.getsize(local_path)
                
                # Basic quality assessment
                quality_score = self._assess_image_quality(img)
                
                # Basic room type classification
                room_type = self._classify_room_type(img, url)
                
                # Determine if exterior/interior
                is_exterior = self._is_exterior_image(img, url)
                
                metadata = ImageMetadata(
                    url=url,
                    filename=filename,
                    width=width,
                    height=height,
                    file_size=file_size,
                    format=format,
                    room_type=room_type,
                    is_exterior=is_exterior,
                    quality_score=quality_score
                )
                
                logger.debug(f"Successfully downloaded image: {filename}")
                return metadata
                
        except Exception as e:
            raise Exception(f"Failed to download image: {e}")
    
    def _assess_image_quality(self, img: Image.Image) -> float:
        """Assess image quality on a scale of 0-1"""
        try:
            # Convert to numpy array for CV operations
            img_array = np.array(img.convert('RGB'))
            
            # Calculate image quality metrics
            
            # 1. Sharpness (Laplacian variance)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            
            # 2. Brightness (avoid too dark/bright images)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Penalize extremes
            
            # 3. Resolution score
            total_pixels = img.width * img.height
            resolution_score = min(1.0, total_pixels / (640 * 480))  # VGA as baseline
            
            # 4. Size requirements
            min_size_score = 1.0 if img.width >= 300 and img.height >= 300 else 0.5
            
            # Combined quality score
            quality_score = (
                sharpness_score * 0.3 +
                brightness_score * 0.2 +
                resolution_score * 0.3 +
                min_size_score * 0.2
            )
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.debug(f"Error assessing image quality: {e}")
            return 0.5  # Default score
    
    def _classify_room_type(self, img: Image.Image, url: str) -> Optional[str]:
        """Basic room type classification using URL hints and simple CV"""
        try:
            # Check URL for hints
            url_lower = url.lower()
            
            # URL-based classification
            if any(word in url_lower for word in ['bedroom', 'chambre', 'bed']):
                return 'bedroom'
            elif any(word in url_lower for word in ['kitchen', 'cuisine']):
                return 'kitchen'
            elif any(word in url_lower for word in ['bathroom', 'salle_bain', 'bath']):
                return 'bathroom'
            elif any(word in url_lower for word in ['living', 'salon', 'sitting']):
                return 'living_room'
            elif any(word in url_lower for word in ['exterior', 'facade', 'outside']):
                return 'exterior'
            
            # Simple color-based heuristics
            img_array = np.array(img.convert('RGB'))
            
            # Calculate dominant colors
            avg_color = np.mean(img_array, axis=(0, 1))
            
            # Green dominant might indicate garden/exterior
            if avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
                return 'garden'
            
            # Blue dominant might indicate bathroom (tiles) or exterior (sky)
            if avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
                return 'bathroom'
            
            return 'unknown'
            
        except Exception as e:
            logger.debug(f"Error classifying room type: {e}")
            return 'unknown'
    
    def _is_exterior_image(self, img: Image.Image, url: str) -> Optional[bool]:
        """Determine if image shows exterior of property"""
        try:
            # Check URL for hints
            url_lower = url.lower()
            exterior_keywords = ['exterior', 'facade', 'outside', 'building', 'front']
            interior_keywords = ['interior', 'inside', 'room', 'kitchen', 'bathroom']
            
            if any(word in url_lower for word in exterior_keywords):
                return True
            elif any(word in url_lower for word in interior_keywords):
                return False
            
            # Simple heuristic: lots of blue/green might indicate sky/vegetation
            img_array = np.array(img.convert('RGB'))
            
            # Calculate color distribution
            blue_ratio = np.mean(img_array[:, :, 2]) / 255.0
            green_ratio = np.mean(img_array[:, :, 1]) / 255.0
            
            if blue_ratio > 0.6 or green_ratio > 0.5:
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error determining interior/exterior: {e}")
            return None


class CloudImageStorage:
    """Handles cloud storage for images (AWS S3 and Google Cloud Storage)"""
    
    def __init__(self, provider: str = "aws", **config):
        self.provider = provider.lower()
        self.config = config
        
        if self.provider == "aws":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config.get('aws_access_key_id'),
                aws_secret_access_key=config.get('aws_secret_access_key'),
                region_name=config.get('region', 'us-east-1')
            )
            self.bucket_name = config.get('bucket_name', 'darvalue-images')
            
        elif self.provider == "gcp":
            self.gcs_client = gcs.Client(
                project=config.get('project_id'),
                credentials=config.get('credentials')
            )
            self.bucket_name = config.get('bucket_name', 'darvalue-images')
        
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")
    
    def upload_image(self, local_path: str, cloud_key: str) -> str:
        """Upload image to cloud storage and return public URL"""
        try:
            if self.provider == "aws":
                return self._upload_to_s3(local_path, cloud_key)
            elif self.provider == "gcp":
                return self._upload_to_gcs(local_path, cloud_key)
            
        except Exception as e:
            logger.error(f"Error uploading image to {self.provider}: {e}")
            raise
    
    def _upload_to_s3(self, local_path: str, key: str) -> str:
        """Upload to AWS S3"""
        try:
            # Upload file
            self.s3_client.upload_file(
                local_path, 
                self.bucket_name, 
                key,
                ExtraArgs={
                    'ContentType': 'image/jpeg',
                    'ACL': 'public-read'
                }
            )
            
            # Return public URL
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{key}"
            return url
            
        except Exception as e:
            raise Exception(f"S3 upload failed: {e}")
    
    def _upload_to_gcs(self, local_path: str, blob_name: str) -> str:
        """Upload to Google Cloud Storage"""
        try:
            bucket = self.gcs_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            # Upload file
            blob.upload_from_filename(local_path)
            
            # Make public
            blob.make_public()
            
            # Return public URL
            return blob.public_url
            
        except Exception as e:
            raise Exception(f"GCS upload failed: {e}")
    
    def batch_upload_images(self, image_metadata_list: List[ImageMetadata], 
                          listing_id: str) -> List[ImageMetadata]:
        """Upload multiple images and update metadata with cloud URLs"""
        updated_metadata = []
        
        for metadata in image_metadata_list:
            try:
                if metadata.download_error:
                    updated_metadata.append(metadata)
                    continue
                
                local_path = os.path.join("data/images", metadata.filename)
                cloud_key = f"listings/{listing_id}/{metadata.filename}"
                
                # Upload to cloud
                cloud_url = self.upload_image(local_path, cloud_key)
                
                # Update metadata
                metadata.storage_path = cloud_url
                updated_metadata.append(metadata)
                
                logger.debug(f"Uploaded {metadata.filename} to cloud storage")
                
            except Exception as e:
                logger.error(f"Error uploading {metadata.filename}: {e}")
                metadata.download_error = f"Upload failed: {e}"
                updated_metadata.append(metadata)
        
        return updated_metadata


class ImageProcessor:
    """Advanced image processing and computer vision features"""
    
    def __init__(self):
        self.image_classifiers = {}
        # In a full implementation, you'd load pre-trained models here
    
    def extract_features(self, image_path: str) -> Dict[str, Any]:
        """Extract computer vision features from image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            features = {}
            
            # 1. Color histogram features
            features['color_histogram'] = self._extract_color_histogram(img)
            
            # 2. Texture features
            features['texture_features'] = self._extract_texture_features(img)
            
            # 3. Edge density
            features['edge_density'] = self._calculate_edge_density(img)
            
            # 4. Brightness and contrast
            features['brightness_contrast'] = self._calculate_brightness_contrast(img)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return {}
    
    def _extract_color_histogram(self, img: np.ndarray) -> List[float]:
        """Extract color histogram features"""
        try:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
            
            # Normalize and flatten
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            # Combine histograms
            combined_hist = np.concatenate([hist_h, hist_s, hist_v])
            return combined_hist.tolist()
            
        except Exception:
            return []
    
    def _extract_texture_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract texture features using simple methods"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate standard deviation (texture roughness)
            texture_roughness = np.std(gray)
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            
            return {
                'roughness': float(texture_roughness),
                'gradient_magnitude': float(avg_gradient)
            }
            
        except Exception:
            return {}
    
    def _calculate_edge_density(self, img: np.ndarray) -> float:
        """Calculate edge density in image"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
            return float(edge_density)
            
        except Exception:
            return 0.0
    
    def _calculate_brightness_contrast(self, img: np.ndarray) -> Dict[str, float]:
        """Calculate brightness and contrast metrics"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            return {
                'brightness': float(brightness / 255.0),
                'contrast': float(contrast / 255.0)
            }
            
        except Exception:
            return {'brightness': 0.5, 'contrast': 0.5}
    
    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect objects in image (placeholder for future ML model)"""
        # Placeholder for object detection
        # In production, you'd use YOLO, R-CNN, or similar models
        
        detected_objects = [
            {'name': 'furniture', 'confidence': 0.8},
            {'name': 'window', 'confidence': 0.9}
        ]
        
        return detected_objects