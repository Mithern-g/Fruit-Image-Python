# ==================== MAIN.PY - REFINED FRUIT GRADING SYSTEM ====================
import Seq_EdgeDetection as edt
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops

# ==================== DIRECTORY CONFIGURATION ====================
FRESH_APPLE_DIR = "train"
ROTTEN_APPLE_DIR = "valid"
test_fresh_dir = "test"
EDGE_ENHANCED_DIR = os.path.join(os.getcwd(), "EdgeEnhanced")
GRADED_OUTPUT_DIR = os.path.join(os.getcwd(), "GradedResults")

# ==================== GLOBAL CONFIGURATION ====================
YOLO_MODEL_PATH = "yolo11n.pt"
YOLO_CONFIG = "yolo11n.yaml"
DATA_CONFIG = os.path.join(os.getcwd(), "data.yaml")

if torch.cuda.is_available():
    DEVICE = 0 
    print(f"[System] GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("[System] No GPU detected, using CPU")

# Set default test image
image_target = "file.jpg"
if os.path.exists(FRESH_APPLE_DIR):
    files = [f for f in os.listdir(FRESH_APPLE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if files:
        image_target = os.path.join(FRESH_APPLE_DIR, files[0])

# ==================== COLOR SPACE TRANSFORMATION MODULE ====================
class ColorSpaceTransformer:
    def __init__(self, target_space='HSV'):
        self.target_space = target_space

    def transform(self, bgr_image, binary_mask):
        """Transforms to HSV and identifies red regions."""
        if self.target_space == 'HSV':
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv_image)
            
            # Red color detection ranges
            lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([179, 255, 255])
            
            mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply binary mask to ensure we only count pixels inside the fruit
            final_red_mask = cv2.bitwise_and(red_mask, binary_mask)
            
            # Calculate red percentage relative to apple size
            apple_pixels = cv2.countNonZero(binary_mask)
            red_pixels = cv2.countNonZero(final_red_mask)
            red_percentage = (red_pixels / apple_pixels * 100) if apple_pixels > 0 else 0
            
            return {
                'color_space': 'HSV',
                'red_percentage': red_percentage,
                'channels': {'H': H, 'S': S, 'V': V},
                'masked_channels': {
                    'H': cv2.bitwise_and(H, H, mask=binary_mask),
                    'S': cv2.bitwise_and(S, S, mask=binary_mask),
                    'V': cv2.bitwise_and(V, V, mask=binary_mask)
                },
                'mask': binary_mask,
                'original_bgr': bgr_image
            }
        return {'color_space': 'UNKNOWN', 'mask': binary_mask, 'original_bgr': bgr_image}

# ==================== FEATURE EXTRACTION MODULE ====================
class FeatureExtractor:
    def extract_features(self, color_data):
        features = {}
        original_bgr = color_data['original_bgr']
        mask = color_data['mask']
        
        # 1. Geometric
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            features['circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        else:
            features['circularity'] = 0
            
        # 2. Texture (GLCM)
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        glcm = graycomatrix(masked_gray, [1], [0], levels=256, symmetric=True, normed=True)
        features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        
        # 3. Color
        features['red_percentage'] = color_data.get('red_percentage', 0)
        
        return features

# ==================== GRADING MODULE ====================
class FruitGrader:
    @staticmethod
    def grade(features):
        red_pct = features.get('red_percentage', 0)
        if red_pct > 80: return 'Grade A - Premium'
        elif red_pct > 50: return 'Grade B - Standard'
        elif red_pct > 20: return 'Grade C - Acceptable'
        else: return 'Reject - Poor Color'

# ==================== GRADING PIPELINE ====================
class FruitGradingPipeline:
    def __init__(self, model, color_space='HSV'):
        self.model = model
        self.color_transformer = ColorSpaceTransformer(target_space=color_space)
        self.feature_extractor = FeatureExtractor()
        self.grader = FruitGrader()

    def create_binary_mask(self, image, bbox):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        rect = (x1, y1, x2-x1, y2-y1)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            return np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        except:
            mask[y1:y2, x1:x2] = 255
            return mask

    def process_detections(self, image_path, yolo_results):
        img = cv2.imread(image_path)
        if img is None: return [], None
        
        results = []
        for result in yolo_results:
            for idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                mask = self.create_binary_mask(img, (x1, y1, x2, y2))
                color_data = self.color_transformer.transform(img, mask)
                features = self.feature_extractor.extract_features(color_data)
                grade = self.grader.grade(features)
                
                results.append({'fruit_index': idx+1, 'features': features, 'grade': grade})
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, grade, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return results, img

# ==================== CORE FUNCTIONS ====================
def Train_Model():
    if not os.path.exists(DATA_CONFIG): return print("data.yaml not found!")
    model = YOLO(YOLO_MODEL_PATH)
    model.train(data=DATA_CONFIG, epochs=15, imgsz=480, device=DEVICE, name="apple_grading_run")
    return model

def Grade_All_Apples():
    """Processes all images in Fresh and Rotten directories and saves results."""
    print("\n" + "="*60)
    print("BATCH GRADING ALL APPLES")
    print("="*60)
    
    # Initialize components
    model = YOLO(YOLO_MODEL_PATH)
    pipeline = FruitGradingPipeline(model, color_space='HSV')
    
    # Define and create output sub-directories
    fresh_out = os.path.join(GRADED_OUTPUT_DIR, "Fresh")
    rotten_out = os.path.join(GRADED_OUTPUT_DIR, "Rotten")
    os.makedirs(fresh_out, exist_ok=True)
    os.makedirs(rotten_out, exist_ok=True)
    
    all_results = []
    # Map the directories to categories
    folders = [('Fresh', FRESH_APPLE_DIR, fresh_out), 
               ('Rotten', ROTTEN_APPLE_DIR, rotten_out)
               ('Mixed', test_fresh_dir, fresh_out),]

    for category, input_dir, output_dir in folders:
        if not os.path.exists(input_dir):
            print(f"✗ Directory not found: {input_dir}")
            continue

        print(f"\nProcessing {category} Apples in: {input_dir}")
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for filename in files:
            img_path = os.path.join(input_dir, filename)
            
            # 1. Run YOLO Detection
            yolo_res = model(img_path, verbose=False)
            
            # 2. Run the Grading Pipeline (Masking -> HSV -> Features -> Grade)
            grading_res, annotated_img = pipeline.process_detections(img_path, yolo_res)
            
            # 3. Save Annotated Image
            if annotated_img is not None:
                cv2.imwrite(os.path.join(output_dir, filename), annotated_img)
            
            # 4. Collect Data for CSV
            for res in grading_res:
                row = {
                    'image': filename, 
                    'actual_category': category, 
                    'calculated_grade': res['grade']
                }
                # Merges numerical features (red_percentage, circularity, etc.) into the row
                row.update(res['features']) 
                all_results.append(row)
                print(f"  ✓ {filename}: {res['grade']}")

    # 5. Save Final CSV Report
    import csv
    csv_path = os.path.join(GRADED_OUTPUT_DIR, 'apple_grading_summary.csv')
    
    if all_results:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            # Uses the keys from the first result as header columns
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        print("\n" + "="*60)
        print(f"✓ Total fruits processed: {len(all_results)}")
        print(f"✓ Summary CSV saved to: {csv_path}")
        print(f"✓ Visual results saved to: {GRADED_OUTPUT_DIR}")
    else:
        print("\n✗ No fruits were detected. CSV was not created.")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("1. Train YOLO | 2. Grade All Apples | 3. Test Single Image")
    choice = input("Select: ")
    if choice == '1': Train_Model()
    elif choice == '2': Grade_All_Apples()
    elif choice == '3':
        path = input("Path: ") or image_target
        model = YOLO(YOLO_MODEL_PATH)
        pipe = FruitGradingPipeline(model, color_space='HSV')
        res, out = pipe.process_detections(path, model(path))
        if out is not None:
            cv2.imshow("Result", out)
            cv2.waitKey(0)