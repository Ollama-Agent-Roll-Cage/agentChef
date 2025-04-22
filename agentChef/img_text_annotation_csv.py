"""img_text_annotation_csv.py

This script processes image files and their associated text annotations,
compiling them into a metadata CSV file. It reads image files and their corresponding text files from a specified input directory,
copies the images to an output directory, and generates a CSV file containing the filenames and their associated text annotations.

You can use this script as a module in your project by importing the `ImageAnnotationProcessor` class as shown below.

```python
from image_annotation_processor import ImageAnnotationProcessor

# Create a processor instance
processor = ImageAnnotationProcessor.create_instance(
    input_dir="path/to/your/input/directory",
    output_dir="path/to/your/output/directory"
)

# Process files and generate the metadata.csv
csv_path = processor.generate_metadata_csv()
print(f"Metadata CSV generated at: {csv_path}")
```

The class is designed to be easily imported from anywhere in your project. It has the following features:

- Handles multiple image formats (png, jpg, jpeg, heic)
- Pairs image files with matching text files
- Error handling for file operations
- Creates the output directory if it doesn't exist
- Generates a metadata.csv with the hugging face img url and text annotation format.

Upload this file to your img annotation dataset folder where your images are located, and this will allow
hugging face to link the images to the text annotations and display them correctly.

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
from typing import List, Dict, Tuple


class ImageAnnotationProcessor:
    """
    A class for processing image files and their associated text annotations,
    compiling them into a metadata CSV file.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the processor with input and output directories.
        
        Args:
            input_dir: Directory containing images and text annotations
            output_dir: Directory where processed files and metadata will be saved
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.heic']
        self.text_extension = '.txt'
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_file_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find all image and text file pairs with matching base names.
        
        Returns:
            List of tuples containing (image_path, text_path)
        """
        file_pairs = []
        
        # Get all files in the input directory
        all_files = list(self.input_dir.glob('*'))
        
        # Separate image and text files
        image_files = [f for f in all_files if f.suffix.lower() in self.image_extensions]
        text_files = [f for f in all_files if f.suffix.lower() == self.text_extension]
        
        # Create a dictionary of text files by their stem (name without extension)
        text_dict = {f.stem: f for f in text_files}
        
        # Find pairs
        for img_file in image_files:
            if img_file.stem in text_dict:
                file_pairs.append((img_file, text_dict[img_file.stem]))
        
        return file_pairs
    
    def _read_text_file(self, file_path: Path) -> str:
        """
        Read and return the contents of a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            String containing the text file contents
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def process_files(self) -> Dict[str, str]:
        """
        Process all file pairs and copy files to output directory.
        
        Returns:
            Dictionary with image filenames as keys and annotation text as values
        """
        metadata = {}
        file_pairs = self._get_file_pairs()
        
        for img_path, txt_path in file_pairs:
            # Read the text annotation
            annotation = self._read_text_file(txt_path)
            
            # Copy the image file to output directory
            try:
                shutil.copy2(img_path, self.output_dir)
                metadata[img_path.name] = annotation
            except Exception as e:
                print(f"Error copying {img_path}: {e}")
        
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
            input_dir: Directory containing images and text annotations
            output_dir: Directory where processed files and metadata will be saved
            
        Returns:
            An instance of ImageAnnotationProcessor
        """
        return ImageAnnotationProcessor(input_dir, output_dir)


# Example usage:
if __name__ == "__main__":
    # Replace with your actual paths
    processor = ImageAnnotationProcessor.create_instance(
        input_dir="./input_images", 
        output_dir="./processed_output"
    )
    
    csv_path = processor.generate_metadata_csv()
    print(f"Metadata CSV generated at: {csv_path}")
