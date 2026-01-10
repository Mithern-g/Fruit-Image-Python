# ==================== SEQ_EDGEDETECTION.PY - REFINED ====================
# Edge Detection Module for Fruit Grading System

import cv2
import sys
import numpy as np
import os

def Edge_Get(source):
    """
    Apply Sobel edge detection and overlay on original image.
    
    This function enhances fruit boundaries for better YOLO detection.
    
    Parameters:
    -----------
    source : str
        Path to input image (supports Windows paths with backslashes)
        
    Returns:
    --------
    numpy.ndarray : Edge-enhanced image, or None if loading fails
    
    Example:
    --------
    edge_img = Edge_Get("C:/Users/jeffy/Fresh1/apple.jpg")
    """
    # Handle Windows paths properly
    source = os.path.normpath(source)
    
    # Read original image
    img = cv2.imread(source)
    
    if img is not None:
        img_height, img_width, channels = img.shape
        
        # Read as grayscale for edge detection
        img_gray = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        
        if img_gray is None:
            print(f"Warning: Could not read image as grayscale: {source}")
            return None

        # Apply Sobel operator for edge detection
        # Horizontal edges (changes in x-direction)
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        
        # Vertical edges (changes in y-direction)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude (combines horizontal and vertical edges)
        gradient_magnitude = cv2.magnitude(sobelx, sobely)

        # Convert to uint8 for display
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
        
        # Initialize output array
        gm = np.zeros(gradient_magnitude.shape, gradient_magnitude.dtype)

        # Edge De-amplification (reduce intensity to 25% for subtle overlay)
        for y in range(gradient_magnitude.shape[0]):
            gm[y] = np.clip(0.25 * gradient_magnitude[y], 0, 255)

        # Convert grayscale edges to RGB for overlay
        gradient_magnitude_2 = cv2.cvtColor(gm, cv2.COLOR_GRAY2RGB)
        
        # Overlay edges on original image using bitwise OR
        img = cv2.bitwise_or(img, gradient_magnitude_2)

        return img
    
    else:
        print(f"Error: Could not load image: {source}")
        return None


def Edge_Get_Batch(input_dir, output_dir):
    """
    Apply edge detection to all images in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input images
    output_dir : str
        Directory to save edge-enhanced images
        
    Returns:
    --------
    int : Number of images processed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    # Supported image formats
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Process all images in directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Apply edge detection
            edge_enhanced = Edge_Get(input_path)
            
            if edge_enhanced is not None:
                cv2.imwrite(output_path, edge_enhanced)
                processed_count += 1
                print(f"✓ Processed: {filename}")
            else:
                print(f"✗ Failed: {filename}")
    
    return processed_count


# ==================== TEST/DEBUG CODE ====================
# This block only runs when you execute this file directly
# It will NOT run when imported by main.py
if __name__ == "__main__":
    print("="*60)
    print("EDGE DETECTION MODULE - TEST MODE")
    print("="*60)
    print("\nImage Naming Convention:")
    print("  Fresh: Fresh_1 to Fresh_40")
    print("  Rotten: Rotten_1 to Rotten_40")
    
    # Test directories (using your actual paths)
    test_fresh_dir = os.path.join("C:\\", "Users", "jeffy", "OneDrive", "Documents", "VisualStudioCode", "DIPGroupAssignment", "Fresh1")
    test_rotten_dir = os.path.join("C:\\", "Users", "jeffy", "OneDrive", "Documents", "VisualStudioCode", "DIPGroupAssignment", "Rotten1")
    
    print("\nSelect test mode:")
    print("1. Test Fresh_1 (First Fresh Apple)")
    print("2. Test Rotten_1 (First Rotten Apple)")
    print("3. Test batch processing All Fresh Apples (Fresh_1 to Fresh_40)")
    print("4. Test batch processing All Rotten Apples (Rotten_1 to Rotten_40)")
    print("5. Test specific image (e.g., Fresh_5 or Rotten_15)")
    print("6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        # Test Fresh_1
        if os.path.exists(test_fresh_dir):
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_image = os.path.join(test_fresh_dir, f"Fresh_1{ext}")
                if os.path.exists(test_image):
                    print(f"\nTesting with: Fresh_1{ext}")
                    result = Edge_Get(test_image)
                    
                    if result is not None:
                        print(f"✓ Edge detection successful")
                        print(f"  Image shape: {result.shape}")
                        
                        # Display the result
                        cv2.imshow("Edge Detection - Fresh_1", result)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                        print(f"✗ Edge detection failed")
                    found = True
                    break
            
            if not found:
                print(f"✗ Fresh_1 not found with any common extension (.jpg, .jpeg, .png, .bmp)")
        else:
            print(f"✗ Directory not found: {test_fresh_dir}")
    
    elif choice == '2':
        # Test Rotten_1
        if os.path.exists(test_rotten_dir):
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_image = os.path.join(test_rotten_dir, f"Rotten_1{ext}")
                if os.path.exists(test_image):
                    print(f"\nTesting with: Rotten_1{ext}")
                    result = Edge_Get(test_image)
                    
                    if result is not None:
                        print(f"✓ Edge detection successful")
                        print(f"  Image shape: {result.shape}")
                        
                        # Display the result
                        cv2.imshow("Edge Detection - Rotten_1", result)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                        print(f"✗ Edge detection failed")
                    found = True
                    break
            
            if not found:
                print(f"✗ Rotten_1 not found with any common extension (.jpg, .jpeg, .png, .bmp)")
        else:
            print(f"✗ Directory not found: {test_rotten_dir}")
    
    elif choice == '3':
        # Batch test fresh apples
        output_dir = r"C:\Users\jeffy\OneDrive\Documents\VisualStudioCode\DIPGroupAssignment\TestOutput\Fresh"
        
        print(f"\nBatch processing Fresh apples...")
        print(f"Input: {test_fresh_dir}")
        print(f"Output: {output_dir}")
        
        if os.path.exists(test_fresh_dir):
            count = Edge_Get_Batch(test_fresh_dir, output_dir)
            print(f"\n✓ Processed {count} fresh apple images")
        else:
            print(f"✗ Directory not found: {test_fresh_dir}")
    
    elif choice == '4':
        # Batch test rotten apples
        output_dir = r"C:\Users\jeffy\OneDrive\Documents\VisualStudioCode\DIPGroupAssignment\TestOutput\Rotten"
        
        print(f"\nBatch processing Rotten apples (Rotten_1 to Rotten_40)...")
        print(f"Input: {test_rotten_dir}")
        print(f"Output: {output_dir}")
        
        if os.path.exists(test_rotten_dir):
            count = Edge_Get_Batch(test_rotten_dir, output_dir)
            print(f"\n✓ Processed {count} rotten apple images")
        else:
            print(f"✗ Directory not found: {test_rotten_dir}")
    
    elif choice == '5':
        # Test specific image
        category = input("\nEnter category (Fresh/Rotten): ").strip().capitalize()
        number = input("Enter number (1-40): ").strip()
        
        if category in ['Fresh', 'Rotten'] and number.isdigit() and 1 <= int(number) <= 40:
            base_dir = test_fresh_dir if category == 'Fresh' else test_rotten_dir
            
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_image = os.path.join(base_dir, f"{category}_{number}{ext}")
                if os.path.exists(test_image):
                    print(f"\nTesting with: {category}_{number}{ext}")
                    result = Edge_Get(test_image)
                    
                    if result is not None:
                        print(f"✓ Edge detection successful")
                        print(f"  Image shape: {result.shape}")
                        
                        # Display the result
                        cv2.imshow(f"Edge Detection - {category}_{number}", result)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                        print(f"✗ Edge detection failed")
                    found = True
                    break
            
            if not found:
                print(f"✗ {category}_{number} not found with any common extension")
        else:
            print("✗ Invalid input. Category must be Fresh/Rotten, number must be 1-40")
    
    elif choice == '6':
        print("\nExiting...")
    
    else:
        print("\n✗ Invalid choice!")
    
    # Exit after testing
    sys.exit(0)