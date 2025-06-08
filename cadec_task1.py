#!/usr/bin/env python3
"""
CADEC Dataset Analysis - Task 1: Entity Enumeration (Final Updated)
Enumerate distinct entities for each label type (ADR, Drug, Disease, Symptom)
Works with actual CADEC file naming: ARTHROTEC.1.ann, CAMBIA.1.ann, etc.
"""

import os
from collections import defaultdict

class CADECEntityEnumerator:
    def __init__(self, cadec_root_dir):
        """
        Initialize with the root directory of CADEC dataset
        
        Args:
            cadec_root_dir (str): Path to the cadec directory
        """
        self.cadec_root = cadec_root_dir
        self.original_dir = os.path.join(cadec_root_dir, 'original')
        self.entities = defaultdict(set)  # {label_type: set of entities}
        
        # Known drug prefixes in CADEC dataset
        self.drug_prefixes = [
            'ARTHROTEC', 'CAMBIA', 'CATAFLAM', 'DICLOFENAC-POTASSIUM',
            'DICLOFENAC-SODIUM', 'FLECTOR', 'LIPITOR', 'PENNSAID',
            'SOLARAZE', 'VOLTAREN', 'VOLTAREN-XR', 'ZIPSOR'
        ]
        
    def parse_original_file(self, filepath):
        """
        Parse a single file from the original subdirectory
        Format: T1	ADR 9 19	bit drowsy
        
        Args:
            filepath (str): Path to the annotation file
            
        Returns:
            list: List of tuples (tag, label, ranges, text)
        """
        annotations = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    
                    # Parse annotation line: T1	ADR 9 19	bit drowsy
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        tag = parts[0]          # T1
                        label_and_ranges = parts[1]  # "ADR 9 19"
                        text = parts[2]         # bit drowsy
                        
                        # Split label and ranges
                        label_parts = label_and_ranges.split()
                        if len(label_parts) >= 3:
                            label = label_parts[0]    # ADR
                            ranges = ' '.join(label_parts[1:])  # "9 19"
                            
                            annotations.append((tag, label, ranges, text))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
                    
        return annotations
    
    def get_all_annotation_files(self):
        """
        Get all annotation files from the original directory
        
        Returns:
            list: List of annotation file paths
        """
        if not os.path.exists(self.original_dir):
            print(f"Error: Original directory not found at {self.original_dir}")
            return []
        
        # Get all .ann files
        all_files = []
        try:
            for filename in os.listdir(self.original_dir):
                if filename.endswith('.ann'):
                    all_files.append(os.path.join(self.original_dir, filename))
        except Exception as e:
            print(f"Error listing files in {self.original_dir}: {e}")
            return []
        
        return sorted(all_files)
    
    def extract_all_entities(self):
        """
        Extract all unique entities from all files in the original directory
        """
        print("Extracting entities from CADEC dataset...")
        
        # Get all annotation files
        annotation_files = self.get_all_annotation_files()
        total_files = len(annotation_files)
        
        if total_files == 0:
            print("No annotation files found!")
            print(f"Please check that files exist in: {self.original_dir}")
            return
        
        print(f"Processing {total_files} annotation files...")
        
        # Show sample of detected files
        sample_files = [os.path.basename(f) for f in annotation_files[:5]]
        print(f"Sample files: {sample_files}")
        
        processed_count = 0
        entity_count = 0
        
        for i, filepath in enumerate(annotation_files):
            filename = os.path.basename(filepath)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{total_files} files...")
                
            try:
                annotations = self.parse_original_file(filepath)
                
                for tag, label, ranges, text in annotations:
                    # Clean and normalize the entity text
                    entity_text = text.strip().lower()
                    
                    # Add to appropriate entity set
                    if label in ['ADR', 'Drug', 'Disease', 'Symptom']:
                        self.entities[label].add(entity_text)
                        entity_count += 1
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"Completed processing {processed_count}/{total_files} files.")
        print(f"Found {entity_count} total entity mentions.")
    
    def get_entity_counts(self):
        """
        Get counts of distinct entities for each label type
        
        Returns:
            dict: Dictionary with label types as keys and counts as values
        """
        return {label: len(entities) for label, entities in self.entities.items()}
    
    def print_summary(self):
        """
        Print summary of distinct entities
        """
        print("\n" + "="*60)
        print("CADEC Dataset - Entity Enumeration Summary")
        print("="*60)
        
        entity_counts = self.get_entity_counts()
        
        for label_type in ['ADR', 'Drug', 'Disease', 'Symptom']:
            if label_type in entity_counts:
                count = entity_counts[label_type]
                print(f"{label_type:10}: {count:6} distinct entities")
            else:
                print(f"{label_type:10}: {0:6} distinct entities")
        
        total_entities = sum(entity_counts.values())
        print(f"{'Total':10}: {total_entities:6} distinct entities")
        print("="*60)
    
    def save_entities_to_files(self, output_dir='entity_lists'):
        """
        Save entity lists to separate files for each label type
        
        Args:
            output_dir (str): Directory to save entity lists
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for label_type, entity_set in self.entities.items():
            # Sort entities alphabetically
            sorted_entities = sorted(list(entity_set))
            
            output_file = os.path.join(output_dir, f'{label_type.lower()}_entities.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                for entity in sorted_entities:
                    f.write(entity + '\n')
            
            print(f"Saved {len(sorted_entities)} {label_type} entities to {output_file}")
    
    def get_sample_entities(self, label_type, n=10):
        """
        Get sample entities for a given label type
        
        Args:
            label_type (str): Type of entity (ADR, Drug, Disease, Symptom)
            n (int): Number of sample entities to return
            
        Returns:
            list: Sample entities
        """
        if label_type in self.entities:
            return list(self.entities[label_type])[:n]
        return []
    
    def get_file_stats(self):
        """Get statistics about files in the dataset"""
        annotation_files = self.get_all_annotation_files()
        
        if not annotation_files:
            return {}
        
        # Count files by drug prefix
        drug_counts = defaultdict(int)
        unmatched_files = []
        
        for filepath in annotation_files:
            filename = os.path.basename(filepath)
            matched = False
            
            for prefix in self.drug_prefixes:
                if filename.startswith(prefix + '.'):
                    drug_counts[prefix] += 1
                    matched = True
                    break
            
            if not matched:
                unmatched_files.append(filename)
        
        stats = dict(drug_counts)
        if unmatched_files:
            stats['_unmatched'] = unmatched_files[:5]  # Show first 5 unmatched
        
        return stats

def main():
    """Main function for Task 1"""
    # Set the path to your CADEC dataset
    cadec_root = './cadec'
    
    if not os.path.exists(cadec_root):
        print(f"Error: CADEC directory not found at {cadec_root}")
        print("Please download and extract CADEC.v2.zip first.")
        print("Extract it so you have: ./cadec/original/, ./cadec/text/, etc.")
        return
    
    # Initialize entity enumerator
    enumerator = CADECEntityEnumerator(cadec_root)
    
    # Show file statistics
    file_stats = enumerator.get_file_stats()
    if file_stats:
        print(f"Found files for drugs:")
        total_files = 0
        for drug, count in file_stats.items():
            if drug != '_unmatched':
                print(f"  {drug}: {count} files")
                total_files += count
        
        if '_unmatched' in file_stats:
            print(f"  Unmatched files: {file_stats['_unmatched']}")
        
        print(f"  Total matched: {total_files} files")
    else:
        print("No annotation files found!")
        return
    
    # Extract all entities
    enumerator.extract_all_entities()
    
    # Print summary
    enumerator.print_summary()
    
    # Save entities to files
    enumerator.save_entities_to_files()
    
    # Show sample entities for each type
    print("\nSample entities:")
    for label_type in ['ADR', 'Drug', 'Disease', 'Symptom']:
        samples = enumerator.get_sample_entities(label_type, 5)
        print(f"\n{label_type} samples:")
        for i, sample in enumerate(samples, 1):
            print(f"  {i}. {sample}")

if __name__ == "__main__":
    main()