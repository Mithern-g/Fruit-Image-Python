# ==================== COMPLETE MODULAR FRUIT GRADING PIPELINE ====================
# This shows the proper separation of each team member's responsibilities

import cv2
import numpy as np
import sys
import os
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO

# ==================== MODULE 1: EDGE DETECTION (Preprocessing) ====================
class EdgeDetector:
    """
    Edge detection preprocessing module.
    Based on Seq_EdgeDetection.py
    """
    
    @staticmethod
    def apply_sobel_edge_detection(image_path):
        """
        Apply Sobel edge detection and overlay on original image.
        
        Parameters:
        -----------
        image_path : str
            Path to input image
            
        Returns:
        --------
        numpy.ndarray : Edge-enhanced image
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Sobel operator
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = cv2.magnitude(sobelx, sobely)
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
        
        # Edge de-amplification
        gm = np.zeros(gradient_magnitude.shape, gradient_magnitude.dtype)
        for y in range(gradient_magnitude.shape[0]):
            gm[y] = np.clip(0.25 * gradient_magnitude[y], 0, 255)
        
        # Overlay on original
        gradient_magnitude_2 = cv2.cvtColor(gm, cv2.COLOR_GRAY2RGB)
        img = cv2.bitwise_or(img, gradient_magnitude_2)
        
        return img


# ==================== MODULE 2: OBJECT IDENTIFICATION (Tan's Role) ====================
class ObjectIdentifier:
    """
    YOLO-based fruit detection and segmentation.
    Tan's responsibility.
    """
    
    def __init__(self, model_path="yolo11n.pt"):
        """Initialize YOLO model."""
        self.model = YOLO(model_path)
        print("[Tan's Module] YOLO model loaded successfully")
    
    def detect_and_segment(self, image_path, conf_threshold=0.4):
        """
        Detect fruits and generate binary masks.
        
        Parameters:
        -----------
        image_path : str
            Path to input image
        conf_threshold : float
            Detection confidence threshold
            
        Returns:
        --------
        list : List of dictionaries containing detection results and masks
        """
        # Load original image
        original_image = cv2.imread(image_path)
        
        # Run YOLO detection
        results = self.model(source=image_path, conf=conf_threshold, verbose=False)
        
        detected_fruits = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Generate binary mask using GrabCut
                binary_mask = self._create_binary_mask(
                    original_image, 
                    (int(x1), int(y1), int(x2), int(y2))
                )
                
                detected_fruits.append({
                    'bounding_box': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'binary_mask': binary_mask,
                    'original_image': original_image
                })
        
        print(f"[Tan's Module] Detected {len(detected_fruits)} fruits")
        return detected_fruits
    
    def _create_binary_mask(self, image, bbox):
        """
        Create refined binary mask from bounding box using GrabCut.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Original image
        bbox : tuple
            Bounding box (x1, y1, x2, y2)
            
        Returns:
        --------
        numpy.ndarray : Binary mask (255=fruit, 0=background)
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        
        # Initialize rectangle mask
        rect = (x1, y1, x2-x1, y2-y1)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut for refinement
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        except:
            # Fallback to simple rectangle mask if GrabCut fails
            binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            binary_mask[y1:y2, x1:x2] = 255
        
        return binary_mask


# ==================== MODULE 3: COLOR SPACE TRANSFORMATION (Ivan's Role) ====================
class ColorSpaceTransformer:
    """
    Color space transformation module.
    Ivan's responsibility - processes between object identification and feature extraction.
    """
    
    def __init__(self, target_space='LAB'):
        """
        Initialize color transformer.
        
        Parameters:
        -----------
        target_space : str
            Target color space ('LAB', 'HSV', 'YCrCb')
        """
        self.target_space = target_space
        print(f"[Ivan's Module] Color transformer initialized: {target_space}")
    
    def transform(self, bgr_image, binary_mask):
        """
        Transform BGR image to target color space.
        
        THIS IS THE KEY STEP: Happens AFTER object identification, BEFORE feature extraction
        
        Parameters:
        -----------
        bgr_image : numpy.ndarray
            Original BGR image
        binary_mask : numpy.ndarray
            Binary mask from object identification
            
        Returns:
        --------
        dict : Transformed image data with separate channels
        """
        if self.target_space == 'LAB':
            return self._transform_to_lab(bgr_image, binary_mask)
        elif self.target_space == 'HSV':
            return self._transform_to_hsv(bgr_image, binary_mask)
        elif self.target_space == 'YCrCb':
            return self._transform_to_ycrcb(bgr_image, binary_mask)
        else:
            raise ValueError(f"Unsupported color space: {self.target_space}")
    
    def _transform_to_lab(self, bgr_image, binary_mask):
        """Transform to CIELAB color space."""
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab_image)
        
        return {
            'color_space': 'LAB',
            'full_image': lab_image,
            'channels': {
                'L': L,  # Lightness
                'a': a,  # Green-Red (ripeness indicator)
                'b': b   # Blue-Yellow
            },
            'masked_channels': {
                'L': cv2.bitwise_and(L, L, mask=binary_mask),
                'a': cv2.bitwise_and(a, a, mask=binary_mask),
                'b': cv2.bitwise_and(b, b, mask=binary_mask)
            },
            'original_bgr': bgr_image,
            'mask': binary_mask
        }
    
    def _transform_to_hsv(self, bgr_image, binary_mask):
        """Transform to HSV color space."""
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv_image)
        
        return {
            'color_space': 'HSV',
            'full_image': hsv_image,
            'channels': {
                'H': H,  # Hue
                'S': S,  # Saturation
                'V': V   # Value
            },
            'masked_channels': {
                'H': cv2.bitwise_and(H, H, mask=binary_mask),
                'S': cv2.bitwise_and(S, S, mask=binary_mask),
                'V': cv2.bitwise_and(V, V, mask=binary_mask)
            },
            'original_bgr': bgr_image,
            'mask': binary_mask
        }
    
    def _transform_to_ycrcb(self, bgr_image, binary_mask):
        """Transform to YCrCb color space."""
        ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb_image)
        
        return {
            'color_space': 'YCrCb',
            'full_image': ycrcb_image,
            'channels': {
                'Y': Y,   # Luminance
                'Cr': Cr, # Red-difference
                'Cb': Cb  # Blue-difference
            },
            'masked_channels': {
                'Y': cv2.bitwise_and(Y, Y, mask=binary_mask),
                'Cr': cv2.bitwise_and(Cr, Cr, mask=binary_mask),
                'Cb': cv2.bitwise_and(Cb, Cb, mask=binary_mask)
            },
            'original_bgr': bgr_image,
            'mask': binary_mask
        }


# ==================== MODULE 4: FEATURE EXTRACTION (Your Role) ====================
class FeatureExtractor:
    """
    Feature extraction module.
    YOUR responsibility - extracts features from color-transformed images.
    """
    
    def __init__(self):
        print("[Your Module] Feature extractor initialized")
    
    def extract_features(self, color_transformed_data):
        """
        Extract geometric, texture, and color features.
        
        THIS IS YOUR MAIN FUNCTION - receives color-transformed data from Ivan
        
        Parameters:
        -----------
        color_transformed_data : dict
            Output from Ivan's ColorSpaceTransformer.transform()
            Contains: color_space, channels, masked_channels, original_bgr, mask
            
        Returns:
        --------
        dict : Extracted features
        """
        features = {}
        
        # Extract necessary data
        original_bgr = color_transformed_data['original_bgr']
        binary_mask = color_transformed_data['mask']
        color_space = color_transformed_data['color_space']
        
        # 1. GEOMETRIC FEATURES
        geo_features = self._extract_geometric_features(binary_mask)
        features.update(geo_features)
        
        # 2. TEXTURE FEATURES
        texture_features = self._extract_texture_features(original_bgr, binary_mask)
        features.update(texture_features)
        
        # 3. COLOR FEATURES (using Ivan's transformed data)
        color_features = self._extract_color_features(color_transformed_data)
        features.update(color_features)
        
        return features
    
    def _extract_geometric_features(self, binary_mask):
        """Extract area, perimeter, and circularity."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            fruit_contour = max(contours, key=cv2.contourArea)
            
            area = cv2.contourArea(fruit_contour)
            perimeter = cv2.arcLength(fruit_contour, closed=True)
            
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0
            
            return {
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity
            }
        
        return {'area': 0, 'perimeter': 0, 'circularity': 0}
    
    def _extract_texture_features(self, bgr_image, binary_mask):
        """Extract GLCM-based texture features."""
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        fruit_gray = cv2.bitwise_and(gray_image, gray_image, mask=binary_mask)
        
        y_coords, x_coords = np.where(binary_mask > 0)
        
        if len(y_coords) > 0:
            min_y, max_y = y_coords.min(), y_coords.max()
            min_x, max_x = x_coords.min(), x_coords.max()
            fruit_region = fruit_gray[min_y:max_y+1, min_x:max_x+1]
            mask_region = binary_mask[min_y:max_y+1, min_x:max_x+1]
            
            fruit_region_masked = fruit_region.copy()
            fruit_region_masked[mask_region == 0] = 0
            
            glcm = graycomatrix(
                fruit_region_masked,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            return {
                'glcm_contrast': graycoprops(glcm, 'contrast')[0, 0],
                'glcm_homogeneity': graycoprops(glcm, 'homogeneity')[0, 0]
            }
        
        return {'glcm_contrast': 0, 'glcm_homogeneity': 0}
    
    def _extract_color_features(self, color_transformed_data):
        """
        Extract color features from Ivan's transformed data.
        
        The key advantage: Ivan has already done the color space conversion,
        so you just extract the statistics from the appropriate channel.
        """
        color_space = color_transformed_data['color_space']
        masked_channels = color_transformed_data['masked_channels']
        mask = color_transformed_data['mask']
        
        features = {}
        
        if color_space == 'LAB':
            # Extract 'a' channel statistics (ripeness indicator)
            a_channel = masked_channels['a']
            a_values = a_channel[mask > 0]
            
            if len(a_values) > 0:
                features['mean_a_channel'] = np.mean(a_values)
                features['std_a_channel'] = np.std(a_values)
            else:
                features['mean_a_channel'] = 0
                features['std_a_channel'] = 0
        
        elif color_space == 'HSV':
            # Extract Hue channel statistics
            h_channel = masked_channels['H']
            h_values = h_channel[mask > 0]
            
            if len(h_values) > 0:
                features['mean_hue'] = np.mean(h_values)
                features['std_hue'] = np.std(h_values)
            else:
                features['mean_hue'] = 0
                features['std_hue'] = 0
        
        return features


# ==================== MODULE 5: GRADING/CLASSIFICATION ====================
class FruitGrader:
    """
    Fruit grading based on extracted features.
    """
    
    @staticmethod
    def grade(features):
        """
        Grade fruit based on features.
        
        Parameters:
        -----------
        features : dict
            Extracted features
            
        Returns:
        --------
        str : Grade ('A', 'B', 'C', 'Reject')
        """
        circularity = features.get('circularity', 0)
        contrast = features.get('glcm_contrast', 0)
        mean_a = features.get('mean_a_channel', 0)
        
        # Grading logic
        if circularity > 0.85 and contrast < 50 and 120 < mean_a < 140:
            return 'Grade A - Premium'
        elif circularity > 0.70 and contrast < 80:
            return 'Grade B - Good'
        elif circularity > 0.60:
            return 'Grade C - Acceptable'
        else:
            return 'Reject - Poor Quality'


# ==================== COMPLETE PIPELINE ORCHESTRATOR ====================
class FruitGradingPipeline:
    """
    Orchestrates the complete pipeline with proper module separation.
    
    Pipeline Flow:
    1. Edge Detection (Preprocessing)
    2. Object Identification (Tan) → Binary Mask
    3. Color Space Transformation (Ivan) → Transformed Channels
    4. Feature Extraction (You) → Feature Vector
    5. Grading → Final Grade
    """
    
    def __init__(self, yolo_model_path="yolo11n.pt", color_space='LAB'):
        """Initialize all modules."""
        self.edge_detector = EdgeDetector()
        self.object_identifier = ObjectIdentifier(yolo_model_path)
        self.color_transformer = ColorSpaceTransformer(color_space)
        self.feature_extractor = FeatureExtractor()
        self.grader = FruitGrader()
        
        print("\n" + "="*60)
        print("FRUIT GRADING PIPELINE INITIALIZED")
        print("="*60)
    
    def process_image(self, image_path, apply_edge_detection=False):
        """
        Process a single image through the complete pipeline.
        
        Parameters:
        -----------
        image_path : str
            Path to input image
        apply_edge_detection : bool
            Whether to apply edge detection preprocessing
            
        Returns:
        --------
        list : Results for each detected fruit
        """
        print(f"\nProcessing: {image_path}")
        print("-" * 60)
        
        # STEP 1: Optional edge detection
        if apply_edge_detection:
            print("Step 1: Applying edge detection...")
            edge_enhanced = self.edge_detector.apply_sobel_edge_detection(image_path)
            # Note: For now, we continue with original for accuracy
        
        # STEP 2: Object Identification (Tan's Module)
        print("Step 2: Object identification (Tan's module)...")
        detected_fruits = self.object_identifier.detect_and_segment(image_path)
        
        if not detected_fruits:
            print("  ✗ No fruits detected")
            return []
        
        results = []
        
        # Process each detected fruit
        for idx, fruit_data in enumerate(detected_fruits):
            print(f"\n--- Processing Fruit #{idx+1} ---")
            
            original_image = fruit_data['original_image']
            binary_mask = fruit_data['binary_mask']
            
            # STEP 3: Color Space Transformation (Ivan's Module)
            print("Step 3: Color space transformation (Ivan's module)...")
            color_data = self.color_transformer.transform(original_image, binary_mask)
            print(f"  ✓ Transformed to {color_data['color_space']}")
            
            # STEP 4: Feature Extraction (Your Module)
            print("Step 4: Feature extraction (Your module)...")
            features = self.feature_extractor.extract_features(color_data)
            print(f"  ✓ Extracted {len(features)} features")
            
            # STEP 5: Grading
            print("Step 5: Grading...")
            grade = self.grader.grade(features)
            print(f"  ✓ Grade: {grade}")
            
            results.append({
                'fruit_index': idx + 1,
                'bounding_box': fruit_data['bounding_box'],
                'confidence': fruit_data['confidence'],
                'features': features,
                'grade': grade
            })
        
        return results
    
    def display_results(self, results):
        """Display formatted results."""
        print("\n" + "="*60)
        print("GRADING RESULTS")
        print("="*60)
        
        for result in results:
            print(f"\nFruit #{result['fruit_index']}:")
            print(f"  Grade: {result['grade']}")
            print(f"  Detection Confidence: {result['confidence']:.2f}")
            print(f"  Features:")
            for key, value in result['features'].items():
                print(f"    - {key}: {value:.4f}")


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = FruitGradingPipeline(
        yolo_model_path="yolo11n.pt",
        color_space='LAB'  # Ivan's choice
    )
    
    # Process single image
    results = pipeline.process_image("fruit.jpg", apply_edge_detection=True)
    
    # Display results
    pipeline.display_results(results)