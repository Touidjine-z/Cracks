"""
Cleanup script: Remove temporary/unnecessary files and keep only essential outputs.
"""

import os
import shutil
from pathlib import Path
import argparse


def cleanup_project(root_dir: str, keep_results: bool = True, verbose: bool = True):
    """Remove unnecessary files from the project."""
    
    cleanup_count = 0
    
    # Files/directories to remove (relative to root)
    remove_items = [
        # Temporary directories
        '__pycache__',
        '.pytest_cache',
        '*.pyc',
        'Cracks-main/.git',
        
        # Intermediate files (if you want to keep only final results)
        'Cracks-main/output/dataset',  # Remove organized dataset (keep annotations)
    ]
    
    # If not keeping results, also remove them
    if not keep_results:
        remove_items.append('Cracks-main/output/results')
    
    for item in remove_items:
        if '*' in item:
            # Glob pattern
            for file in Path(root_dir).rglob(item.replace('*', '*')):
                if file.is_file() and file.suffix in ['.pyc']:
                    if verbose:
                        print(f"Removing: {file}")
                    os.remove(file)
                    cleanup_count += 1
        else:
            full_path = os.path.join(root_dir, item)
            if os.path.exists(full_path):
                if os.path.isdir(full_path):
                    if verbose:
                        print(f"Removing directory: {full_path}")
                    shutil.rmtree(full_path)
                else:
                    if verbose:
                        print(f"Removing file: {full_path}")
                    os.remove(full_path)
                cleanup_count += 1
    
    return cleanup_count


def print_final_structure(root_dir: str):
    """Print the final project structure."""
    print("\n" + "=" * 70)
    print("FINAL PROJECT STRUCTURE")
    print("=" * 70)
    
    cracks_main = os.path.join(root_dir, 'Cracks-main')
    
    def tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = []
        try:
            items = sorted(os.listdir(directory))
        except PermissionError:
            return
        
        # Filter
        items = [i for i in items if not i.startswith('.')]
        
        for i, item in enumerate(items):
            path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item}")
            
            if os.path.isdir(path) and current_depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                tree(path, prefix + extension, max_depth, current_depth + 1)
    
    print(f"{cracks_main}/")
    tree(cracks_main)


def main():
    parser = argparse.ArgumentParser(description="Cleanup project and remove unnecessary files")
    parser.add_argument("--root", default=".", help="Root directory")
    parser.add_argument("--keep-dataset", action="store_true", help="Keep organized dataset (default: remove)")
    parser.add_argument("--remove-results", action="store_true", help="Also remove inference results")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PROJECT CLEANUP")
    print("=" * 70)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE: No files will be deleted\n")
    
    count = cleanup_project(
        args.root,
        keep_results=not args.remove_results,
        verbose=not args.quiet
    )
    
    print(f"\n✓ Cleanup complete: {count} items removed" if count > 0 else "\n✓ No items to remove")
    
    print_final_structure(args.root)
    
    print("\n" + "=" * 70)
    print("📂 FINAL STRUCTURE")
    print("=" * 70)
    print("""
Cracks-main/
├── dataset/
│   ├── positive/       (Original positive crack images)
│   └── negative/       (Optional negative images)
├── output/
│   ├── annotations/    (Auto-generated YOLO labels - 20K .txt files)
│   ├── model/          (Model config + data.yaml for YOLOv8 training)
│   ├── results/        (Inference outputs: masks + bbox images - 40K files)
│   └── dataset/        (REMOVED - Use Kaggle dataset directly if needed)
├── PIPELINE_DOCUMENTATION.md
└── README.md

Root level scripts:
├── auto_annotate.py        (Step 1: Generate annotations)
├── prepare_dataset.py      (Step 2: Organize for training)
├── test_mask_inference.py  (Step 3: Run inference)
├── pipeline.py             (Main orchestrator)
└── detect_mask.py          (Segmentation algorithm)
""")
    
    print("\n✅ Project ready for use!")


if __name__ == "__main__":
    main()
