# ==================== MAIN.PY - REFINED FRUIT GRADING SYSTEM ====================
from unittest import result
import Seq_EdgeDetection as edt
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops

# ==================== DIRECTORY CONFIGURATION ====================
FRESH_APPLE_DIR = os.path.join("train", "images")
ROTTEN_APPLE_DIR = os.path.join("valid", "images")
test_fresh_dir = os.path.join("finaltest")
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
        if self.target_space == 'HSV':
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            apple_area = cv2.countNonZero(binary_mask)
            
            # 1. RED MASK (Fresh)
            lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
            red_mask = cv2.bitwise_or(cv2.inRange(hsv_image, lower_red1, upper_red1),
                                     cv2.inRange(hsv_image, lower_red2, upper_red2))

            # 2. YELLOW MASK (Fresh/Variety)
            lower_yellow, upper_yellow = np.array([18, 50, 100]), np.array([32, 255, 255])
            yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

            # 3. GREEN MASK (Fresh/Unripe) - New Feature
            lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

            # 4. BROWN/BLACK MASK (Rotten)
            #lower_brown, upper_brown = np.array([5, 40, 10]), np.array([17, 255, 120]) Replaced for testing reasons
            lower_brown, upper_brown = np.array([1, 135, 0]), np.array([21, 255, 150]) #For Reddish/Brownish Dark Colors
            brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

            # 5. Decay Mask (Addition)
            lower_decay, upper_decay = np.array([8, 0, 165]), np.array([10, 115, 255]) #For funny fungi color
            decay_mask = cv2.inRange(hsv_image, lower_decay, upper_decay)

            # 6. Oddity Mask (Addition)
            lower_oddity, upper_oddity = np.array([16,205,215]), np.array([18,255,255]) #Odd spots on some of the bad ones
            oddity_mask = cv2.inRange(hsv_image, lower_oddity, upper_oddity)

            # Calculate Percentages
            red_pct    = (cv2.countNonZero(cv2.bitwise_and(red_mask, binary_mask)) / apple_area * 100) if apple_area > 0 else 0
            yellow_pct = (cv2.countNonZero(cv2.bitwise_and(yellow_mask, binary_mask)) / apple_area * 100) if apple_area > 0 else 0
            green_pct  = (cv2.countNonZero(cv2.bitwise_and(green_mask, binary_mask)) / apple_area * 100) if apple_area > 0 else 0
            brown_pct  = (cv2.countNonZero(cv2.bitwise_and(brown_mask, binary_mask)) / apple_area * 100) if apple_area > 0 else 0
            decay_pct  = (cv2.countNonZero(cv2.bitwise_and(decay_mask, binary_mask)) / apple_area * 100) if apple_area > 0 else 0
            oddity_pct = (cv2.countNonZero(cv2.bitwise_and(oddity_mask, binary_mask)) / apple_area * 100) if apple_area > 0 else 0

            return {
                'color_space': 'HSV',
                'red_percentage': red_pct,
                'yellow_percentage': yellow_pct,
                'green_percentage': green_pct,
                'brown_percentage': brown_pct,
                'decay_percentage': decay_pct,
                'oddity_percentage': oddity_pct,
                'mask': binary_mask,
                'original_bgr': bgr_image
            }

# ==================== FEATURE EXTRACTION MODULE ====================
class FeatureExtractor:
    def extract_features(self, color_data):
        features = {}
        original_bgr = color_data['original_bgr']
        mask = color_data['mask']
        
        # 1. GEOMETRIC FIX
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            # Ensure this is saved to the 'features' dictionary
            features['circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        else:
            features['circularity'] = 0
            
        # 2. TEXTURE FIX (GLCM)
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        glcm = graycomatrix(masked_gray, [1], [0], levels=256, symmetric=True, normed=True)
        # Ensure this is saved to the 'features' dictionary
        features['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
        
        # 3. COLOR PASS-THROUGH
        features['red_percentage'] = color_data.get('red_percentage', 0)
        features['yellow_percentage'] = color_data.get('yellow_percentage', 0)
        features['green_percentage'] = color_data.get('green_percentage', 0)
        features['brown_percentage'] = color_data.get('brown_percentage', 0)
        features['oddity_percentage'] = color_data.get('oddity_percentage', 0)
        features['decay_percentage'] = color_data.get('decay_percentage', 0)

        return features

# ==================== GRADING MODULE ====================
class FruitGrader:
    @staticmethod
    def grade(features):
        # 1. Feature Extraction
        red = features.get('red_percentage', 0)
        yellow = features.get('yellow_percentage', 0)
        green = features.get('green_percentage', 0)
        brown = features.get('brown_percentage', 0)
        decay = features.get('decay_percentage', 0)
        oddity = features.get('oddity_percentage',0)
        contrast = features.get('glcm_contrast', 0)
        circularity = features.get('circularity', 0)

        #Adjust as one desire, this isn't accurate, after all.'
        brown_weight = 1 #Higher due to capability of marking deep red/brown bad situations
        oddity_weight = 80.0 #Lower due to inaccuracy and weird situations
        decay_weight = 4.0 #Less due to I am not very confident in this.

        # 2. Competitive Scoring
        fresh_score = red + yellow + green
        awful_score = brown*brown_weight + oddity*oddity_weight + decay*decay_weight

        # 3. RULE: Structural/Texture Failure (Highest Priority)
        # Fixes msg5170347760-69603 (Circ: 0.39) and msg5170347760-71081 (Contrast: 34)
        #if 0 < circularity < 0.60 or contrast > 33.5:
        #    return 'Rotten'

        # 4. RULE: Color Dominance (Freshness Shield)
        # Fixes 6100533069082640056 (Brown: 21.7% but Yellow: 47.5%)
        #if brown > 12.0:
        #    if fresh_score > (brown * 1.4):
        #        return 'Fresh'
        #    else:
        #        return 'Rotten'

        #4. Rule: Freshness scoring (Experimental)
        if(awful_score > 30): #Summary of color is beyond 20, which either means 20% is weird, or the design is wrong.
            return 'Rotten' #On further thought, let's ignore freshness, since if it seems rotten, it might as well not be fresh.

        # 5. RULE: Combined Decay (Rough + Non-circular)
        #if contrast > 28.0 and circularity < 0.82:
        #    return 'Rotten'

        return 'Fresh'
    
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
            if len(result.boxes) > 0:
                box = result.boxes[0] 
                idx = 0
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
    model.train(data=DATA_CONFIG, epochs=30, imgsz=480, device=DEVICE, name="apple_grading_run")
    return model

def Grade_All_Apples():
    """Processes all images in Fresh and Rotten directories and saves results."""
    # 15 images in total for testing
    print("\n" + "="*15)
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
    '''
            (Fresh', FRESH_APPLE_DIR, fresh_out), 
            ('Rotten', ROTTEN_APPLE_DIR, rotten_out),
            ('Test', test_fresh_dir, fresh_out)
    '''
    # Map the directories to categories
    folders = [('Fresh', FRESH_APPLE_DIR, fresh_out)]

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
                    'calculated_grade': res['grade'],
                    # Captures the AI's confidence in identifying the fruit
                    'yolo_conf': round(float(yolo_res[0].boxes.conf[0]), 2) if len(yolo_res[0].boxes) > 0 else 0
                }
                
                # Merges all numerical features into the row:
                # red_pct, yellow_pct, green_pct, brown_pct, circularity, glcm_contrast
                row.update(res['features']) 
                all_results.append(row)
                print(f"   ✓ {filename}: {res['grade']}")

    # 5. Save Final CSV Report
    import csv
    csv_path = os.path.join(GRADED_OUTPUT_DIR, 'apple_grading_summary.csv')
    
    
    if all_results:
        # Define the exact order you want in Excel
        # In your CSV writing section:
        fieldnames = [
            'image', 'actual_category', 'calculated_grade', 
            'red_percentage', 'yellow_percentage', 'green_percentage', 'brown_percentage', 'decay_percentage', 'oddity_percentage',
            'circularity', 'glcm_contrast'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            # Use extrasaction='ignore' to prevent errors if a key is missing
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
        
        print("\n" + "="*60)
        print(f"✓ Total fruits processed: {len(all_results)}")
        print(f"✓ Summary CSV saved to: {csv_path}")
        print(f"✓ Visual results saved to: {GRADED_OUTPUT_DIR}")
    else:
        print("\n✗ No fruits were detected. CSV was not created.")

    # Initialize counters before the loop
    total_count = 0
    correct_count = 0

    # ... Inside your processing loop ...
    for res in grading_res:
        total_count += 1
        # Compare AI result with the Actual Label from folder/filename
        if res['grade'].lower() == category.lower():
            correct_count += 1
        
        # Existing collection logic
        row = {'image': filename, 'actual': category, 'predicted': res['grade']}
        row.update(res['features'])
        all_results.append(row)

    # After the loop finishes, calculate and print the summary
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("\n" + "="*40)
        print(f"FINAL PERFORMANCE SUMMARY")
        print(f"Total Apples Processed: {total_count}")
        print(f"Correct Classifications: {correct_count}")
        print(f"SYSTEM ACCURACY: {accuracy:.2f}%")
        print("="*40 + "\n")

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
