"""
Batch processing and evaluation utilities for crack detection
"""

import cv2
import numpy as np
import os
from pathlib import Path
from crack_detector_cv import CrackDetectorCV
import time
from tqdm import tqdm


def batch_detect_and_visualize(image_dir, detector, output_dir='detections', 
                                confidence_threshold=0.5, max_images=None):
    """Detect cracks in multiple images and save visualizations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(Path(image_dir).glob('*.jpg'))
    if max_images:
        image_files = image_files[:max_images]
    
    results = []
    total_time = 0
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for img_file in tqdm(image_files):
        start = time.time()
        
        img, detections = detector.detect_cracks(
            str(img_file), 
            confidence_threshold=confidence_threshold
        )
        
        elapsed = time.time() - start
        total_time += elapsed
        
        if img is not None:
            # Save visualization
            vis = detector.visualize_detections(
                img, detections,
                output_path=os.path.join(output_dir, f"{img_file.stem}_detected.jpg")
            )
            
            results.append({
                'image': img_file.name,
                'detections': len(detections),
                'time': elapsed,
                'boxes': detections
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Total detections: {sum(r['detections'] for r in results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per image: {total_time/len(results):.2f}s")
    print(f"Images with cracks: {sum(1 for r in results if r['detections'] > 0)}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return results


def evaluate_speed(detector, num_images=10):
    """Benchmark inference speed"""
    
    print(f"\nBenchmarking inference speed ({num_images} images)...")
    
    # Create dummy images
    times = []
    for i in range(num_images):
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        temp_file = f'/tmp/test_img_{i}.jpg'
        cv2.imwrite(temp_file, img)
        
        start = time.time()
        _, _ = detector.detect_cracks(temp_file, confidence_threshold=0.5)
        elapsed = time.time() - start
        
        times.append(elapsed)
        os.remove(temp_file)
    
    print(f"\nInference Speed Benchmark:")
    print(f"  Average: {np.mean(times):.3f}s")
    print(f"  Min: {np.min(times):.3f}s")
    print(f"  Max: {np.max(times):.3f}s")
    print(f"  Std: {np.std(times):.3f}s")
    print(f"  FPS: {1/np.mean(times):.1f}")


def generate_report(results, output_file='detection_report.txt'):
    """Generate detailed detection report"""
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CRACK DETECTION REPORT\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"Image: {result['image']}\n")
            f.write(f"  Cracks detected: {result['detections']}\n")
            f.write(f"  Processing time: {result['time']:.3f}s\n")
            
            if result['boxes']:
                f.write(f"  Bounding boxes:\n")
                for i, box in enumerate(result['boxes']):
                    f.write(f"    {i+1}. Position: ({box['x']}, {box['y']}) "
                           f"Size: {box['width']}x{box['height']} "
                           f"Confidence: {box['confidence']:.3f}\n")
            f.write("\n")
        
        # Statistics
        total_detections = sum(r['detections'] for r in results)
        avg_time = np.mean([r['time'] for r in results])
        
        f.write("="*60 + "\n")
        f.write("STATISTICS\n")
        f.write("="*60 + "\n")
        f.write(f"Total images: {len(results)}\n")
        f.write(f"Total cracks: {total_detections}\n")
        f.write(f"Average detections per image: {total_detections/len(results):.2f}\n")
        f.write(f"Average processing time: {avg_time:.3f}s\n")
        f.write(f"Images with cracks: {sum(1 for r in results if r['detections'] > 0)}\n")
    
    print(f"\nReport saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch processing utilities')
    parser.add_argument('--batch', type=str, help='Batch detect images in directory')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark inference speed')
    parser.add_argument('--max-images', type=int, default=50, help='Max images to process')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='detections', help='Output directory')
    
    args = parser.parse_args()
    
    detector = CrackDetectorCV()
    detector.load_model()
    
    if args.batch:
        results = batch_detect_and_visualize(
            args.batch, detector, args.output, 
            args.confidence, args.max_images
        )
        generate_report(results, os.path.join(args.output, 'report.txt'))
    
    if args.benchmark:
        evaluate_speed(detector, args.max_images)


if __name__ == '__main__':
    main()
