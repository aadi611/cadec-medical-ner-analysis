#!/usr/bin/env python3
"""
CADEC Dataset Analysis - Task 3: Performance Evaluation (Final Updated)
Evaluate NER predictions against ground truth using overlap-based metrics
Works with actual CADEC file format: T1	ADR 9 19	bit drowsy
"""

import os
from collections import defaultdict

class CADECEvaluator:
    def __init__(self, cadec_root):
        """
        Initialize the evaluator
        
        Args:
            cadec_root (str): Path to CADEC dataset root directory
        """
        self.cadec_root = cadec_root
        self.original_dir = os.path.join(cadec_root, 'original')
        self.meddra_dir = os.path.join(cadec_root, 'meddra')
        self.text_dir = os.path.join(cadec_root, 'text')
    
    def parse_annotation_file(self, filepath, filter_label=None):
        """
        Parse an annotation file and return entities
        Format: T1	ADR 9 19	bit drowsy
        
        Args:
            filepath (str): Path to annotation file
            filter_label (str): Only return entities with this label (optional)
            
        Returns:
            list: List of entity tuples (start, end, label, text)
        """
        entities = []
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found - {filepath}")
            return entities
        
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
                        tag = parts[0]              # T1
                        label_and_ranges = parts[1] # "ADR 9 19"
                        text = parts[2]             # bit drowsy
                        
                        # Split label and ranges
                        label_parts = label_and_ranges.split()
                        if len(label_parts) >= 3:
                            label = label_parts[0]    # ADR
                            ranges = ' '.join(label_parts[1:])  # "9 19"
                            
                            # Filter by label if specified
                            if filter_label and label != filter_label:
                                continue
                            
                            # Parse ranges (can be multiple ranges separated by ';')
                            for range_str in ranges.split(';'):
                                range_parts = range_str.strip().split()
                                if len(range_parts) >= 2:
                                    try:
                                        start = int(range_parts[0])
                                        end = int(range_parts[1])
                                        entities.append((start, end, label, text))
                                    except ValueError:
                                        continue
        
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
        
        return entities
    
    def parse_meddra_file(self, filepath):
        """
        Parse MedDRA annotation file
        Format: TT2	10019326 0 9	heartburn
        
        Args:
            filepath (str): Path to MedDRA annotation file
            
        Returns:
            list: List of entity tuples (start, end, label, text)
        """
        entities = []
        
        if not os.path.exists(filepath):
            return entities
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    # Parse MedDRA line: TT2	10019326 0 9	heartburn
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        tag = parts[0]              # TT2
                        code_and_ranges = parts[1]  # "10019326 0 9"
                        text = parts[2]             # heartburn
                        
                        # Split code and ranges: "10019326 0 9"
                        code_parts = code_and_ranges.split()
                        if len(code_parts) >= 3:
                            # Format: [MedDRA_code] [start] [end]
                            meddra_code = code_parts[0]  # 10019326
                            start = int(code_parts[1])   # 0
                            end = int(code_parts[2])     # 9
                            
                            # All MedDRA annotations are ADR by definition
                            entities.append((start, end, 'ADR', text))
        
        except Exception as e:
            print(f"Error parsing MedDRA file {filepath}: {e}")
        
        return entities
    
    def parse_prediction_file(self, filepath):
        """
        Parse prediction file (same format as original)
        Format: T1	ADR 9 19	bit drowsy
        
        Args:
            filepath (str): Path to prediction file
            
        Returns:
            list: List of entity tuples (start, end, label, text)
        """
        entities = []
        
        if not os.path.exists(filepath):
            return entities
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    # Parse prediction line: T1	ADR 9 19	bit drowsy
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        tag = parts[0]              # T1
                        label_and_ranges = parts[1] # "ADR 9 19"
                        text = parts[2]             # bit drowsy
                        
                        # Split label and ranges
                        label_parts = label_and_ranges.split()
                        if len(label_parts) >= 3:
                            label = label_parts[0]    # ADR
                            ranges = ' '.join(label_parts[1:])  # "9 19"
                            
                            # Parse ranges
                            range_parts = ranges.split()
                            if len(range_parts) >= 2:
                                try:
                                    start = int(range_parts[0])
                                    end = int(range_parts[1])
                                    entities.append((start, end, label, text))
                                except ValueError:
                                    continue
        
        except Exception as e:
            print(f"Error parsing prediction file {filepath}: {e}")
        
        return entities
    
    def calculate_overlap(self, pred_span, true_span):
        """
        Calculate overlap between two spans
        
        Args:
            pred_span (tuple): (start, end) of predicted span
            true_span (tuple): (start, end) of true span
            
        Returns:
            float: Overlap ratio (0-1)
        """
        pred_start, pred_end = pred_span
        true_start, true_end = true_span
        
        # Calculate intersection
        intersection_start = max(pred_start, true_start)
        intersection_end = min(pred_end, true_end)
        
        if intersection_start >= intersection_end:
            return 0.0
        
        intersection_length = intersection_end - intersection_start
        union_length = max(pred_end, true_end) - min(pred_start, true_start)
        
        return intersection_length / union_length if union_length > 0 else 0.0
    
    def overlap_based_evaluation(self, predictions, truth, min_overlap=0.5):
        """
        Evaluate using overlap-based criteria (more lenient than exact match)
        
        JUSTIFICATION FOR OVERLAP-BASED EVALUATION:
        
        We chose overlap-based evaluation over exact match for the following reasons:
        
        1. **Robustness to Boundary Differences**: Medical text often has ambiguous 
           entity boundaries. For example, "blurred vision" vs "little blurred vision" 
           should both be considered correct ADR identifications even if boundaries differ.
        
        2. **Realistic Performance Measurement**: Exact match is too strict for medical NER.
           If a model identifies "stomach irritation" as positions 5-22 but ground truth 
           is 4-23, it's still a successful identification of the medical concept.
        
        3. **Clinical Relevance**: In medical applications, identifying the right concept
           is more important than perfect character-level boundaries. A doctor would 
           consider both "drowsy" and "bit drowsy" as identifying the same ADR.
        
        4. **Handles Tokenization Differences**: Different models may tokenize text 
           differently, leading to slight boundary variations that don't affect 
           semantic meaning.
        
        5. **Standard in Medical NER**: Overlap-based evaluation (often with IoU > 0.5) 
           is commonly used in biomedical NER research as it better reflects real-world 
           application needs.
        
        Alternative methods considered:
        - Exact match: Too strict, penalizes minor boundary differences
        - Token-level F1: Good but doesn't handle multi-word entities well
        - BLEU/ROUGE: Designed for text generation, not entity recognition
        
        Args:
            predictions (list): List of predicted entities (start, end, label, text)
            truth (list): List of ground truth entities (start, end, label, text)
            min_overlap (float): Minimum overlap threshold for considering a match (0.5 = 50% IoU)
            
        Returns:
            dict: Evaluation metrics per label and overall
        """
        results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Group by label
        pred_by_label = defaultdict(list)
        true_by_label = defaultdict(list)
        
        for entity in predictions:
            pred_by_label[entity[2]].append(entity)
        
        for entity in truth:
            true_by_label[entity[2]].append(entity)
        
        # Get all labels
        all_labels = set(pred_by_label.keys()) | set(true_by_label.keys())
        
        for label in all_labels:
            pred_entities = pred_by_label[label]
            true_entities = true_by_label[label]
            
            # Track which true entities have been matched
            matched_true = set()
            
            # For each prediction, find best matching true entity
            for pred_entity in pred_entities:
                pred_span = (pred_entity[0], pred_entity[1])
                best_overlap = 0
                best_true_idx = -1
                
                for i, true_entity in enumerate(true_entities):
                    if i in matched_true:
                        continue
                    
                    true_span = (true_entity[0], true_entity[1])
                    overlap = self.calculate_overlap(pred_span, true_span)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_true_idx = i
                
                # Check if overlap meets threshold
                if best_overlap >= min_overlap:
                    results[label]['tp'] += 1
                    matched_true.add(best_true_idx)
                else:
                    results[label]['fp'] += 1
            
            # Count unmatched true entities as false negatives
            results[label]['fn'] = len(true_entities) - len(matched_true)
        
        # Calculate metrics for each label
        metrics = {}
        overall_tp, overall_fp, overall_fn = 0, 0, 0
        
        for label in all_labels:
            tp = results[label]['tp']
            fp = results[label]['fp']
            fn = results[label]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
            
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
        
        # Calculate overall metrics
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'support': overall_tp + overall_fn
        }
        
        return metrics
    
    def evaluate_single_file(self, filename, prediction_dir, use_meddra=False):
        """
        Evaluate predictions for a single file against ground truth
        
        Args:
            filename (str): Name of the file (without extension)
            prediction_dir (str): Directory containing prediction files
            use_meddra (bool): Use meddra directory instead of original for ADR evaluation
            
        Returns:
            dict: Evaluation results
        """
        # Load ground truth
        if use_meddra:
            truth_file = os.path.join(self.meddra_dir, filename + '.ann')
            truth_entities = self.parse_meddra_file(truth_file)
        else:
            truth_file = os.path.join(self.original_dir, filename + '.ann')
            truth_entities = self.parse_annotation_file(truth_file)
        
        # Load predictions
        pred_file = os.path.join(prediction_dir, filename + '.ann')
        pred_entities = self.parse_prediction_file(pred_file)
        
        # Perform evaluation using overlap-based method
        results = self.overlap_based_evaluation(pred_entities, truth_entities)
        
        return results
    
    def print_evaluation_results(self, results, title="Evaluation Results"):
        """
        Print formatted evaluation results
        
        Args:
            results (dict): Evaluation results dictionary
            title (str): Title for the results
        """
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        # Print per-label results
        for label in ['ADR', 'Drug', 'Disease', 'Symptom']:
            if label in results:
                metrics = results[label]
                print(f"{label:10} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | Support: {metrics['support']}")
        
        # Print overall results
        if 'overall' in results:
            overall = results['overall']
            print(f"{'Overall':10} | P: {overall['precision']:.3f} | R: {overall['recall']:.3f} | F1: {overall['f1']:.3f} | Support: {overall['support']}")
        
        print(f"{'='*60}")

def demo_evaluation():
    """
    Demonstrate evaluation on a sample file
    """
    cadec_root = './cadec'
    prediction_dir = './predictions'
    
    if not os.path.exists(cadec_root):
        print(f"Error: CADEC directory not found at {cadec_root}")
        return
    
    if not os.path.exists(prediction_dir):
        print(f"Error: Predictions directory not found at {prediction_dir}")
        print("Please run Task 2 first to generate predictions.")
        return
    
    evaluator = CADECEvaluator(cadec_root)
    
    # Find first available prediction file
    pred_files = [f for f in os.listdir(prediction_dir) if f.endswith('.ann')]
    if not pred_files:
        print("No prediction files found!")
        print("Please run Task 2 first to generate predictions.")
        return
    
    filename = pred_files[0].replace('.ann', '')
    
    print(f"Evaluating predictions for {filename}")
    
    # Standard evaluation (using original annotations)
    results_standard = evaluator.evaluate_single_file(filename, prediction_dir, use_meddra=False)
    evaluator.print_evaluation_results(results_standard, "Standard Evaluation (Original Annotations)")
    
    # ADR-specific evaluation (using meddra annotations)
    results_adr = evaluator.evaluate_single_file(filename, prediction_dir, use_meddra=True)
    evaluator.print_evaluation_results(results_adr, "ADR-Specific Evaluation (MedDRA Annotations)")

def main():
    """Main function for Task 3"""
    demo_evaluation()

if __name__ == "__main__":
    main()