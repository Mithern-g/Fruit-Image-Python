# ==================== MAIN.PY - REFINED FOR YOUR DIRECTORY STRUCTURE ====================
import Seq_EdgeDetection as edt
import os
import cv2
import numpy as np
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops

# ==================== DIRECTORY CONFIGURATION ====================
# Your specific directories - using os.path.join to avoid backslash issues
FRESH_APPLE_DIR = os.path.join("C:\\", "Users", "jeffy", "OneDrive", "Documents", "VisualStudioCode", "DIPGroupAssignment", "Fresh1")
ROTTEN_APPLE_DIR = os.path.join("C:\\", "Users", "jeffy", "OneDrive", "Documents", "VisualStudioCode", "DIPGroupAssignment", "Rotten1")

# Output directories
EDGE_ENHANCED_DIR = os.path.join("C:\\", "Users", "jeffy", "OneDrive", "Documents", "VisualStudioCode", "DIPGroupAssignment", "EdgeEnhanced")
GRADED_OUTPUT_DIR = os.path.join("C:\\", "Users", "jeffy", "OneDrive", "Documents", "VisualStudioCode", "DIPGroupAssignment", "GradedResults")

# ==================== GLOBAL CONFIGURATION ====================
# YOLO configuration
YOLO_MODEL_PATH = "yolo11n.pt"
YOLO_CONFIG = "yolo11n.yaml"
DATA_CONFIG = "data.yaml"

# Auto-detect device (GPU if available, otherwise CPU)
import torch
if torch.cuda.is_available():
    DEVICE = 0  # Use first GPU
    print(f"[System] GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("[System] No GPU detected, using CPU")

# Test image - defaults to Fresh_1 (your naming convention: Fresh_1 to Fresh_40)
if os.path.exists(FRESH_APPLE_DIR):
    # Try to find Fresh_1 with common extensions
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        test_path = os.path.join(FRESH_APPLE_DIR, f"Fresh_1{ext}")
        if os.path.exists(test_path):
            image_target = test_path
            break
    else:
        # Fallback to first file in directory
        files = [f for f in os.listdir(FRESH_APPLE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_target = os.path.join(FRESH_APPLE_DIR, files[0]) if files else "file.jpg"
else:
    image_target = "file.jpg"

# ==================== COLOR SPACE TRANSFORMATION MODULE (Ivan's Role) ====================
class ColorSpaceTransformer:
    """Ivan's module: Transforms color space after object identification."""
    
    def __init__(self, target_space='LAB'):
        self.target_space = target_space
        print(f"[Ivan's Module] Color transformer initialized: {target_space}")
    
    def transform(self, bgr_image, binary_mask):
        """Transform BGR image to target color space."""
        if self.target_space == 'LAB':
            lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
            L, a, b = cv2.split(lab_image)
            
            return {
                'color_space': 'LAB',
                'full_image': lab_image,
                'channels': {'L': L, 'a': a, 'b': b},
                'masked_channels': {
                    'L': cv2.bitwise_and(L, L, mask=binary_mask),
                    'a': cv2.bitwise_and(a, a, mask=binary_mask),
                    'b': cv2.bitwise_and(b, b, mask=binary_mask)
                },
                'original_bgr': bgr_image,
                'mask': binary_mask
            }
        
        elif self.target_space == 'HSV':
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv_image)
            
            return {
                'color_space': 'HSV',
                'full_image': hsv_image,
                'channels': {'H': H, 'S': S, 'V': V},
                'masked_channels': {
                    'H': cv2.bitwise_and(H, H, mask=binary_mask),
                    'S': cv2.bitwise_and(S, S, mask=binary_mask),
                    'V': cv2.bitwise_and(V, V, mask=binary_mask)
                },
                'original_bgr': bgr_image,
                'mask': binary_mask
            }


# ==================== FEATURE EXTRACTION MODULE (Your Role) ====================
class FeatureExtractor:
    """Your module: Extracts geometric, texture, and color features."""
    
    def __init__(self):
        print("[Your Module] Feature extractor initialized")
    
    def extract_features(self, color_transformed_data):
        """Extract features from color-transformed data."""
        features = {}
        
        original_bgr = color_transformed_data['original_bgr']
        binary_mask = color_transformed_data['mask']
        
        # 1. Geometric Features
        geo_features = self._extract_geometric_features(binary_mask)
        features.update(geo_features)
        
        # 2. Texture Features
        texture_features = self._extract_texture_features(original_bgr, binary_mask)
        features.update(texture_features)
        
        # 3. Color Features
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
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            return {'area': area, 'perimeter': perimeter, 'circularity': circularity}
        
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
            
            glcm = graycomatrix(fruit_region_masked, distances=[1], angles=[0], 
                               levels=256, symmetric=True, normed=True)
            
            return {
                'glcm_contrast': graycoprops(glcm, 'contrast')[0, 0],
                'glcm_homogeneity': graycoprops(glcm, 'homogeneity')[0, 0]
            }
        
        return {'glcm_contrast': 0, 'glcm_homogeneity': 0}
    
    def _extract_color_features(self, color_transformed_data):
        """Extract color features from transformed data."""
        color_space = color_transformed_data['color_space']
        masked_channels = color_transformed_data['masked_channels']
        mask = color_transformed_data['mask']
        
        features = {}
        
        if color_space == 'LAB':
            a_channel = masked_channels['a']
            a_values = a_channel[mask > 0]
            
            if len(a_values) > 0:
                features['mean_a_channel'] = np.mean(a_values)
                features['std_a_channel'] = np.std(a_values)
            else:
                features['mean_a_channel'] = 0
                features['std_a_channel'] = 0
        
        elif color_space == 'HSV':
            h_channel = masked_channels['H']
            h_values = h_channel[mask > 0]
            
            if len(h_values) > 0:
                features['mean_hue'] = np.mean(h_values)
                features['std_hue'] = np.std(h_values)
            else:
                features['mean_hue'] = 0
                features['std_hue'] = 0
        
        return features


# ==================== GRADING MODULE ====================
class FruitGrader:
    """Grade fruits based on extracted features."""
    
    @staticmethod
    def grade(features):
        """Grade fruit based on features."""
        circularity = features.get('circularity', 0)
        contrast = features.get('glcm_contrast', 0)
        mean_a = features.get('mean_a_channel', 0)
        
        # Grading criteria (adjust based on your dataset)
        if circularity > 0.85 and contrast < 50 and 120 < mean_a < 140:
            return 'Grade A - Premium'
        elif circularity > 0.70 and contrast < 80:
            return 'Grade B - Good'
        elif circularity > 0.60:
            return 'Grade C - Acceptable'
        else:
            return 'Reject - Poor Quality'


# ==================== GRADING PIPELINE ====================
class FruitGradingPipeline:
    """Complete grading pipeline."""
    
    def __init__(self, model, color_space='LAB'):
        self.model = model
        self.color_transformer = ColorSpaceTransformer(color_space)
        self.feature_extractor = FeatureExtractor()
        self.grader = FruitGrader()
        print("[Pipeline] Grading pipeline initialized")
    
    def create_binary_mask(self, image, bbox):
        """Create binary mask from bounding box using GrabCut."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        
        rect = (x1, y1, x2-x1, y2-y1)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        except:
            binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            binary_mask[y1:y2, x1:x2] = 255
        
        return binary_mask
    
    def process_detections(self, image_path, yolo_results):
        """Process YOLO detections and extract features + grade."""
        original_image = cv2.imread(image_path)
        if original_image is None:
            return [], None
        
        all_results = []
        
        for result in yolo_results:
            boxes = result.boxes
            
            for idx, box in enumerate(boxes):
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])
                
                # Create binary mask (Tan's module output)
                binary_mask = self.create_binary_mask(original_image, (x1, y1, x2, y2))
                
                # Color space transformation (Ivan's module)
                color_data = self.color_transformer.transform(original_image, binary_mask)
                
                # Feature extraction (Your module)
                features = self.feature_extractor.extract_features(color_data)
                
                # Grading
                grade = self.grader.grade(features)
                
                all_results.append({
                    'fruit_index': idx + 1,
                    'bounding_box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'features': features,
                    'grade': grade
                })
                
                # Draw on image
                cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_image, grade, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return all_results, original_image


# ==================== TRAINING FUNCTION ====================
def Train_Model():
    """YOLO training function with auto GPU/CPU selection."""
    print("\n" + "="*60)
    print("YOLO MODEL TRAINING")
    print("="*60)
    
    # Display device info
    if DEVICE == 'cpu':
        print("⚠️  Training on CPU (slow)")
        print("   To use GPU, install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print(f"✓ Training on GPU: {torch.cuda.get_device_name(0)}")
    
    # Load a model
    model = YOLO(YOLO_CONFIG)
    print("✓ Model architecture loaded")
    
    model = YOLO(YOLO_MODEL_PATH)
    print("✓ Pretrained weights loaded")
    
    model = YOLO(YOLO_CONFIG).load(YOLO_MODEL_PATH)
    print("✓ Weights transferred")

    # Train the model with auto-detected device
    print("\nStarting training...")
    print(f"Device: {DEVICE}")
    print("Epochs: 100")
    print("Image size: 480x480")
    
    model.train(data=DATA_CONFIG, epochs=100, imgsz=480, device=DEVICE)
    print("✓ Training complete")
    
    return model


# ==================== EDGE-ENHANCED DATASET GENERATION ====================
def Generate_Edge_Enhanced_Dataset():
    """
    Generate edge-enhanced dataset for both Fresh and Rotten apples.
    """
    print("\n" + "="*60)
    print("GENERATING EDGE-ENHANCED DATASET")
    print("="*60)
    
    # Create output directories
    fresh_output = os.path.join(EDGE_ENHANCED_DIR, "Fresh")
    rotten_output = os.path.join(EDGE_ENHANCED_DIR, "Rotten")
    
    os.makedirs(fresh_output, exist_ok=True)
    os.makedirs(rotten_output, exist_ok=True)
    
    # Process Fresh apples
    print("\nProcessing Fresh apples...")
    if os.path.exists(FRESH_APPLE_DIR):
        count = 0
        for filename in os.listdir(FRESH_APPLE_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(FRESH_APPLE_DIR, filename)
                edge_enhanced = edt.Edge_Get(image_path)
                
                if edge_enhanced is not None:
                    output_path = os.path.join(fresh_output, filename)
                    cv2.imwrite(output_path, edge_enhanced)
                    count += 1
                    print(f"  ✓ Processed: {filename}")
        print(f"  Total Fresh apples processed: {count}")
    else:
        print(f"  ✗ Directory not found: {FRESH_APPLE_DIR}")
    
    # Process Rotten apples
    print("\nProcessing Rotten apples...")
    if os.path.exists(ROTTEN_APPLE_DIR):
        count = 0
        for filename in os.listdir(ROTTEN_APPLE_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(ROTTEN_APPLE_DIR, filename)
                edge_enhanced = edt.Edge_Get(image_path)
                
                if edge_enhanced is not None:
                    output_path = os.path.join(rotten_output, filename)
                    cv2.imwrite(output_path, edge_enhanced)
                    count += 1
                    print(f"  ✓ Processed: {filename}")
        print(f"  Total Rotten apples processed: {count}")
    else:
        print(f"  ✗ Directory not found: {ROTTEN_APPLE_DIR}")
    
    print(f"\n✓ Edge-enhanced dataset saved to: {EDGE_ENHANCED_DIR}")


# ==================== PROCESS SINGLE IMAGE ====================
def Process_Single_Image(image_path=None, with_grading=True):
    """
    Process a single image with optional grading.
    
    Parameters:
    -----------
    image_path : str, optional
        Path to image. If None, uses image_target
    with_grading : bool
        If True, performs complete grading pipeline
    """
    if image_path is None:
        image_path = image_target
    
    print(f"\n" + "="*60)
    print(f"PROCESSING IMAGE: {os.path.basename(image_path)}")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"✗ Error: File not found: {image_path}")
        return None
    
    # Apply edge detection
    print("\nStep 1: Applying edge detection...")
    processed = edt.Edge_Get(image_path)
    
    # Load YOLO model
    print("Step 2: Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH)
    
    # Run YOLO detection
    print("Step 3: Running YOLO detection...")
    result = model(source=image_path, show=False, conf=0.4, save=False, verbose=False)
    
    if with_grading:
        print("Step 4: Running grading pipeline...")
        
        # Initialize grading pipeline
        pipeline = FruitGradingPipeline(model, color_space='LAB')
        
        # Process detections with grading
        grading_results, annotated_image = pipeline.process_detections(image_path, result)
        
        # Display results
        print("\n" + "="*60)
        print("GRADING RESULTS")
        print("="*60)
        
        if grading_results:
            for res in grading_results:
                print(f"\nFruit #{res['fruit_index']}:")
                print(f"  Grade: {res['grade']}")
                print(f"  Detection Confidence: {res['confidence']:.2f}")
                print(f"  Features:")
                for key, value in res['features'].items():
                    print(f"    - {key}: {value:.4f}")
            
            # Show annotated image
            cv2.imshow("Grading Results", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("\n✗ No fruits detected in the image")
        
        return grading_results
    else:
        # Original behavior - just show YOLO detection
        result = model(source=image_path, show=True, conf=0.4, save=True, verbose=False)
        return result


# ==================== BATCH GRADING FUNCTION ====================
def Grade_All_Apples():
    """
    Grade all apples in both Fresh and Rotten directories.
    """
    print("\n" + "="*60)
    print("BATCH GRADING ALL APPLES")
    print("="*60)
    
    # Load model
    print("\nInitializing YOLO model...")
    model = YOLO(YOLO_MODEL_PATH)
    pipeline = FruitGradingPipeline(model, color_space='LAB')
    
    # Create output directories
    fresh_graded = os.path.join(GRADED_OUTPUT_DIR, "Fresh")
    rotten_graded = os.path.join(GRADED_OUTPUT_DIR, "Rotten")
    
    os.makedirs(fresh_graded, exist_ok=True)
    os.makedirs(rotten_graded, exist_ok=True)
    
    all_results = []
    
    # Process Fresh apples
    print("\n" + "-"*60)
    print("Processing Fresh Apples")
    print("-"*60)
    
    if os.path.exists(FRESH_APPLE_DIR):
        for filename in os.listdir(FRESH_APPLE_DIR):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            image_path = os.path.join(FRESH_APPLE_DIR, filename)
            
            # Run YOLO detection
            yolo_results = model(source=image_path, conf=0.4, save=False, verbose=False)
            
            # Process with grading
            grading_results, annotated_image = pipeline.process_detections(image_path, yolo_results)
            
            # Save annotated image
            output_path = os.path.join(fresh_graded, filename)
            if annotated_image is not None:
                cv2.imwrite(output_path, annotated_image)
            
            # Store results
            for res in grading_results:
                all_results.append({
                    'image': filename,
                    'category': 'Fresh',
                    'actual_quality': 'Fresh',
                    'grade': res['grade'],
                    'features': res['features']
                })
                
                print(f"✓ {filename}: {res['grade']}")
    
    # Process Rotten apples
    print("\n" + "-"*60)
    print("Processing Rotten Apples")
    print("-"*60)
    
    if os.path.exists(ROTTEN_APPLE_DIR):
        for filename in os.listdir(ROTTEN_APPLE_DIR):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            image_path = os.path.join(ROTTEN_APPLE_DIR, filename)
            
            # Run YOLO detection
            yolo_results = model(source=image_path, conf=0.4, save=False, verbose=False)
            
            # Process with grading
            grading_results, annotated_image = pipeline.process_detections(image_path, yolo_results)
            
            # Save annotated image
            output_path = os.path.join(rotten_graded, filename)
            if annotated_image is not None:
                cv2.imwrite(output_path, annotated_image)
            
            # Store results
            for res in grading_results:
                all_results.append({
                    'image': filename,
                    'category': 'Rotten',
                    'actual_quality': 'Rotten',
                    'grade': res['grade'],
                    'features': res['features']
                })
                
                print(f"✓ {filename}: {res['grade']}")
    
    # Save results to CSV
    import csv
    csv_path = os.path.join(GRADED_OUTPUT_DIR, 'apple_grading_summary.csv')
    
    if all_results:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image', 'category', 'actual_quality', 'grade'] + list(all_results[0]['features'].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_results:
                row = {
                    'image': result['image'],
                    'category': result['category'],
                    'actual_quality': result['actual_quality'],
                    'grade': result['grade']
                }
                row.update(result['features'])
                writer.writerow(row)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"✓ Total apples graded: {len(all_results)}")
        print(f"✓ Fresh apples: {sum(1 for r in all_results if r['category'] == 'Fresh')}")
        print(f"✓ Rotten apples: {sum(1 for r in all_results if r['category'] == 'Rotten')}")
        print(f"✓ Results saved to: {csv_path}")
        print(f"✓ Annotated images saved to: {GRADED_OUTPUT_DIR}")
    
    return all_results


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FRUIT GRADING SYSTEM - APPLE QUALITY ASSESSMENT")
    print("="*60)
    print(f"\nFresh Apple Directory: {FRESH_APPLE_DIR}")
    print(f"Rotten Apple Directory: {ROTTEN_APPLE_DIR}")
    print(f"\nNaming Convention:")
    print(f"  Fresh: Fresh_1 to Fresh_40")
    print(f"  Rotten: Rotten_1 to Rotten_40")
    
    # Check if directories exist
    fresh_exists = os.path.exists(FRESH_APPLE_DIR)
    rotten_exists = os.path.exists(ROTTEN_APPLE_DIR)
    
    print(f"\nFresh directory status: {'✓ Found' if fresh_exists else '✗ Not found'}")
    print(f"Rotten directory status: {'✓ Found' if rotten_exists else '✗ Not found'}")
    
    if fresh_exists:
        fresh_count = len([f for f in os.listdir(FRESH_APPLE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"  Fresh apples: {fresh_count} images")
    
    if rotten_exists:
        rotten_count = len([f for f in os.listdir(ROTTEN_APPLE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"  Rotten apples: {rotten_count} images")
    
    print("\n" + "="*60)
    print("Select operation:")
    print("1. Train YOLO Model")
    print("2. Generate Edge-Enhanced Dataset (Fresh_1 to Fresh_40 + Rotten_1 to Rotten_40)")
    print("3. Process Single Image - YOLO Only")
    print("4. Process Single Image - With Grading")
    print("5. Grade All 80 Apples (Complete Pipeline)")
    print("6. Test on Fresh_1 (First Fresh Apple)")
    print("7. Test on Rotten_1 (First Rotten Apple)")
    print("8. Test on Specific Image (e.g., Fresh_5 or Rotten_15)")
    
    choice = input("\nEnter choice (1-8): ").strip()
    
    if choice == '1':
        # Train YOLO model
        Train_Model()
    
    elif choice == '2':
        # Generate edge-enhanced dataset
        Generate_Edge_Enhanced_Dataset()
    
    elif choice == '3':
        # Process single image - YOLO only
        img_path = input("Enter image path (or press Enter for Fresh_1): ").strip()
        if not img_path:
            img_path = image_target
        Process_Single_Image(img_path, with_grading=False)
    
    elif choice == '4':
        # Process single image with grading
        img_path = input("Enter image path (or press Enter for Fresh_1): ").strip()
        if not img_path:
            img_path = image_target
        Process_Single_Image(img_path, with_grading=True)
    
    elif choice == '5':
        # Grade all apples
        Grade_All_Apples()
    
    elif choice == '6':
        # Test on Fresh_1
        if fresh_exists:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_image = os.path.join(FRESH_APPLE_DIR, f"Fresh_1{ext}")
                if os.path.exists(test_image):
                    print(f"\nTesting with: Fresh_1{ext}")
                    Process_Single_Image(test_image, with_grading=True)
                    break
            else:
                print("✗ Fresh_1 not found. Available extensions: .jpg, .jpeg, .png, .bmp")
        else:
            print("✗ Fresh apple directory not found")
    
    elif choice == '7':
        # Test on Rotten_1
        if rotten_exists:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_image = os.path.join(ROTTEN_APPLE_DIR, f"Rotten_1{ext}")
                if os.path.exists(test_image):
                    print(f"\nTesting with: Rotten_1{ext}")
                    Process_Single_Image(test_image, with_grading=True)
                    break
            else:
                print("✗ Rotten_1 not found. Available extensions: .jpg, .jpeg, .png, .bmp")
        else:
            print("✗ Rotten apple directory not found")
    
    elif choice == '8':
        # Test on specific image by number
        category = input("Enter category (Fresh/Rotten): ").strip().capitalize()
        number = input("Enter number (1-40): ").strip()
        
        if category in ['Fresh', 'Rotten'] and number.isdigit() and 1 <= int(number) <= 40:
            base_dir = FRESH_APPLE_DIR if category == 'Fresh' else ROTTEN_APPLE_DIR
            
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_image = os.path.join(base_dir, f"{category}_{number}{ext}")
                if os.path.exists(test_image):
                    print(f"\nTesting with: {category}_{number}{ext}")
                    Process_Single_Image(test_image, with_grading=True)
                    break
            else:
                print(f"✗ {category}_{number} not found with any common extension")
        else:
            print("✗ Invalid input. Use: Fresh/Rotten and number 1-40")
    
    else:
        print("Invalid choice!")