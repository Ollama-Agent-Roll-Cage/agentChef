"""img_annotation_processor.py

This script processes image files and their associated text annotations,
compiling them into a metadata CSV file. It can either use annotations provided directly as a dictionary
or process images from a directory and generate a CSV file.

You can use this script as a module in your project by importing the `ImageAnnotationProcessor` class as shown below.

```python
from img_annotation_processor import ImageAnnotationProcessor

# Example 1: Create a processor with dictionary annotations
annotations = {
    "image1.jpg": "A beautiful sunset over mountains",
    "image2.png": "A cat playing with a toy"
}

processor = ImageAnnotationProcessor.create_with_annotations(
    input_dir="path/to/your/images",
    output_dir="path/to/your/output/directory",
    annotations=annotations
)

# Process files and generate the metadata.csv
csv_path = processor.generate_metadata_csv()
print(f"Metadata CSV generated at: {csv_path}")

# Example 2: Just process images from a directory
processor = ImageAnnotationProcessor.create_instance(
    input_dir="path/to/your/input/directory",
    output_dir="path/to/your/output/directory"
)

# You can add annotations later
processor.add_annotations({
    "image3.jpg": "A scenic landscape with a river",
    "image4.png": "Portrait of a smiling person"
})

csv_path = processor.generate_metadata_csv()
print(f"Metadata CSV generated at: {csv_path}")
```

The class is designed to be easily integrated with other annotation generators. It has the following features:

- Handles multiple image formats (png, jpg, jpeg, heic)
- Accepts annotations directly as dictionaries
- Error handling for file operations
- Creates the output directory if it doesn't exist
- Generates a metadata.csv with the hugging face img url and text annotation format

This allows you to build separate annotation generators that can feed into this processor.

Example hugging face metadata.csv file:
```url
# my fractal LoRA dataset annotations contains a metadata.csv for this purpose
https://huggingface.co/datasets/Borcherding/HuggingFaceIcons-imageAnnotations-v0.1/blob/main/train/metadata.csv
```
"""

import os
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Optional


class ImageAnnotationProcessor:
    """
    A class for processing image files and their associated text annotations,
    compiling them into a metadata CSV file.
    """
    
    def __init__(self, input_dir: str, output_dir: str, annotations: Optional[Dict[str, str]] = None):
        """
        Initialize the processor with input and output directories and optional annotations.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory where processed files and metadata will be saved
            annotations: Optional dictionary with image filenames as keys and annotation text as values
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.heic']
        self.annotations = annotations or {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_annotations(self, annotations: Dict[str, str]) -> None:
        """
        Add or update annotations for images.
        
        Args:
            annotations: Dictionary with image filenames as keys and annotation text as values
        """
        self.annotations.update(annotations)
    
    def _get_image_files(self) -> List[Path]:
        """
        Find all image files in the input directory.
        
        Returns:
            List of paths to image files
        """
        all_files = list(self.input_dir.glob('*'))
        return [f for f in all_files if f.suffix.lower() in self.image_extensions]
    
    def process_files(self) -> Dict[str, str]:
        """
        Process all image files and copy them to output directory.
        If annotations are provided, they are used; otherwise, empty annotations are created.
        
        Returns:
            Dictionary with image filenames as keys and annotation text as values
        """
        metadata = {}
        image_files = self._get_image_files()
        
        for img_path in image_files:
            # Copy the image file to output directory
            try:
                shutil.copy2(img_path, self.output_dir)
                # Use provided annotation if available, otherwise use empty string
                metadata[img_path.name] = self.annotations.get(img_path.name, "")
            except Exception as e:
                print(f"Error copying {img_path}: {e}")
        
        # Also include annotations for files that might not be in the input directory
        # but are specified in the annotations dictionary
        for filename, annotation in self.annotations.items():
            if filename not in metadata:
                # Check if the file exists in the output directory already
                output_file = self.output_dir / filename
                if output_file.exists():
                    metadata[filename] = annotation
                else:
                    print(f"Warning: Annotation provided for {filename} but file not found in input directory")
        
        return metadata
    
    def generate_metadata_csv(self) -> str:
        """
        Process files and generate metadata CSV file.
        
        Returns:
            Path to the generated CSV file
        """
        metadata = self.process_files()
        csv_path = self.output_dir / 'metadata.csv'
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['file_name', 'text'])
                
                # Write data rows
                for filename, text in metadata.items():
                    writer.writerow([filename, text])
            
            return str(csv_path)
        except Exception as e:
            print(f"Error creating CSV file: {e}")
            return ""

    @staticmethod
    def create_instance(input_dir: str, output_dir: str) -> 'ImageAnnotationProcessor':
        """
        Factory method to create and return an instance of the processor.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory where processed files and metadata will be saved
            
        Returns:
            An instance of ImageAnnotationProcessor
        """
        return ImageAnnotationProcessor(input_dir, output_dir)
    
    @staticmethod
    def create_with_annotations(input_dir: str, output_dir: str, 
                              annotations: Dict[str, str]) -> 'ImageAnnotationProcessor':
        """
        Factory method to create and return an instance of the processor with annotations.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory where processed files and metadata will be saved
            annotations: Dictionary with image filenames as keys and annotation text as values
            
        Returns:
            An instance of ImageAnnotationProcessor with annotations
        """
        return ImageAnnotationProcessor(input_dir, output_dir, annotations)


# Example usage:
if __name__ == "__main__":
    # Example with direct annotations
    sample_annotations = {
        "image1.jpg": "A scenic mountain landscape",
        "image2.png": "A busy city street at night"
    }
    
    processor = ImageAnnotationProcessor.create_with_annotations(
        input_dir="./input_images", 
        output_dir="./processed_output",
        annotations=sample_annotations
    )
    
    csv_path = processor.generate_metadata_csv()
    print(f"Metadata CSV generated at: {csv_path}")
