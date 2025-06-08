#!/usr/bin/env python3
"""
CADEC Dataset Analysis - Task 5: Batch Evaluation on 50 Random Files
Use your code in 3 to measure performance on 50 randomly selected forum
posts from sub-directory text.
"""

import os
import random
import pandas as pd
from cadec_task2 import MedicalNERTagger
from cadec_task3 import CADECEvaluator
import time
import statistics

def get_50_random_files(cadec_root, seed=42):
    """
    Get 50 random text files from the dataset
    
    Args:
        cadec_root (str): Path to CADEC dataset root
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of 50 selected filenames (without extension)
    """
    text_dir = os.path.join(cadec_root, 'text')
    
    if not os.path.exists(text_dir):
        print(f"Error: Text directory not found at {text_dir}")
        return []
    
    # Get all text files
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    
    if len(text_files) == 0:
        print("No text files found!")
        return []
    
    print(f"Found {len(text_files)} total text files in dataset")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Select exactly 50 random files (or all available if less than 50)
    n_files = min(50, len(text_files))
    selected_files = random.sample(text_files, n_files)
    
    print(f"Randomly selected {n_files} files for evaluation")
    
    # Return without extension
    return [f.replace('.txt', '') for f in selected_files]

def process_single_file_task5(filename, cadec_root, tagger, evaluator):
    """
    Process a single file using Task 3 methodology
    
    Args:
        filename (str): Filename without extension
        cadec_root (str): CADEC root directory
        tagger: Medical NER tagger instance
        evaluator: CADEC evaluator instance
        
    Returns:
        tuple: (evaluation_results, processing_time, success)
    """
    start_time = time.time()
    
    # Generate predictions
    text_filepath = os.path.join(cadec_root, 'text', filename + '.txt')
    
    try:
        if not os.path.exists(text_filepath):
            print(f"Warning: Text file not found - {text_filepath}")
            return None, time.time() - start_time, False
        
        # Process the text file using Task 2 methodology
        original_text, bio_tagged_words, annotations = tagger.process_text_file(text_filepath)
        
        # Save predictions
        predictions_dir = './task5_predictions'
        os.makedirs(predictions_dir, exist_ok=True)
        pred_filepath = os.path.join(predictions_dir, filename + '.ann')
        tagger.save_annotations(annotations, pred_filepath)
        
        # Evaluate predictions using Task 3 methodology
        # Using original annotations as ground truth (same as Task 3)
        evaluation_results = evaluator.evaluate_single_file(filename, predictions_dir, use_meddra=False)
        
        processing_time = time.time() - start_time
        
        return evaluation_results, processing_time, True
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, time.time() - start_time, False

def calculate_batch_statistics(all_results):
    """
    Calculate statistics across all files
    
    Args:
        all_results (list): List of evaluation results for each file
        
    Returns:
        dict: Comprehensive statistics
    """
    if not all_results:
        return {}
    
    stats = {}
    
    # Collect metrics for each label type
    for label in ['ADR', 'Drug', 'Disease', 'Symptom', 'overall']:
        precision_values = []
        recall_values = []
        f1_values = []
        support_values = []
        
        for result in all_results:
            if label in result['metrics']:
                metrics = result['metrics'][label]
                precision_values.append(metrics['precision'])
                recall_values.append(metrics['recall'])
                f1_values.append(metrics['f1'])
                support_values.append(metrics['support'])
        
        if precision_values:  # Only calculate if we have data
            stats[label] = {
                'precision': {
                    'mean': statistics.mean(precision_values),
                    'median': statistics.median(precision_values),
                    'std': statistics.stdev(precision_values) if len(precision_values) > 1 else 0,
                    'min': min(precision_values),
                    'max': max(precision_values)
                },
                'recall': {
                    'mean': statistics.mean(recall_values),
                    'median': statistics.median(recall_values),
                    'std': statistics.stdev(recall_values) if len(recall_values) > 1 else 0,
                    'min': min(recall_values),
                    'max': max(recall_values)
                },
                'f1': {
                    'mean': statistics.mean(f1_values),
                    'median': statistics.median(f1_values),
                    'std': statistics.stdev(f1_values) if len(f1_values) > 1 else 0,
                    'min': min(f1_values),
                    'max': max(f1_values)
                },
                'support': {
                    'total': sum(support_values),
                    'mean': statistics.mean(support_values),
                    'median': statistics.median(support_values)
                }
            }
    
    return stats

def save_detailed_results(all_results, output_dir='./task5_results'):
    """
    Save detailed results to CSV file
    
    Args:
        all_results (list): List of all evaluation results
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    
    for result in all_results:
        row = {
            'filename': result['filename'],
            'processing_time': result['processing_time']
        }
        
        # Add metrics for each label
        for label in ['ADR', 'Drug', 'Disease', 'Symptom', 'overall']:
            if label in result['metrics']:
                metrics = result['metrics'][label]
                row[f'{label}_precision'] = metrics['precision']
                row[f'{label}_recall'] = metrics['recall']
                row[f'{label}_f1'] = metrics['f1']
                row[f'{label}_support'] = metrics['support']
            else:
                row[f'{label}_precision'] = 0.0
                row[f'{label}_recall'] = 0.0
                row[f'{label}_f1'] = 0.0
                row[f'{label}_support'] = 0
        
        csv_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    csv_file = os.path.join(output_dir, 'task5_detailed_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to: {csv_file}")

def print_comprehensive_summary(stats, n_files, total_time):
    """
    Print comprehensive summary of Task 5 results
    
    Args:
        stats (dict): Calculated statistics
        n_files (int): Number of files processed
        total_time (float): Total processing time
    """
    print(f"\n{'='*80}")
    print(f"TASK 5 COMPREHENSIVE RESULTS: PERFORMANCE ON {n_files} RANDOM FILES")
    print(f"{'='*80}")
    print(f"Methodology: Same as Task 3 (overlap-based evaluation, original ground truth)")
    print(f"Random seed: 42 (for reproducibility)")
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"Average time per file: {total_time / n_files:.2f} seconds")
    print(f"{'='*80}")
    
    # Overall performance summary
    if 'overall' in stats:
        overall_stats = stats['overall']
        print(f"\nOVERALL PERFORMANCE SUMMARY:")
        print(f"{'='*40}")
        print(f"F1-Score:  {overall_stats['f1']['mean']:.3f} ± {overall_stats['f1']['std']:.3f}")
        print(f"  Range:   {overall_stats['f1']['min']:.3f} - {overall_stats['f1']['max']:.3f}")
        print(f"  Median:  {overall_stats['f1']['median']:.3f}")
        print(f"")
        print(f"Precision: {overall_stats['precision']['mean']:.3f} ± {overall_stats['precision']['std']:.3f}")
        print(f"Recall:    {overall_stats['recall']['mean']:.3f} ± {overall_stats['recall']['std']:.3f}")
        print(f"")
        print(f"Total entities in ground truth: {overall_stats['support']['total']}")
        print(f"Average entities per file: {overall_stats['support']['mean']:.1f}")
    
    # Per-label performance
    print(f"\nPER-LABEL PERFORMANCE:")
    print(f"{'='*40}")
    for label in ['ADR', 'Drug', 'Disease', 'Symptom']:
        if label in stats:
            label_stats = stats[label]
            print(f"{label:8} F1: {label_stats['f1']['mean']:.3f} ± {label_stats['f1']['std']:.3f} "
                  f"(Support: {label_stats['support']['total']})")
    
    # Performance distribution
    if 'overall' in stats:
        overall_f1 = stats['overall']['f1']
        print(f"\nPERFORMACE DISTRIBUTION (F1-Scores):")
        print(f"{'='*40}")
        print(f"Best performing file:  {overall_f1['max']:.3f}")
        print(f"Worst performing file: {overall_f1['min']:.3f}")
        print(f"Standard deviation:    {overall_f1['std']:.3f}")
        
        # Performance categories
        mean_f1 = overall_f1['mean']
        if mean_f1 > 0.7:
            performance_category = "EXCELLENT"
        elif mean_f1 > 0.5:
            performance_category = "GOOD"
        elif mean_f1 > 0.3:
            performance_category = "MODERATE"
        else:
            performance_category = "NEEDS IMPROVEMENT"
        
        print(f"Average performance:   {performance_category}")
    
    print(f"{'='*80}")

def main():
    """
    Main function for Task 5
    """
    print("="*60)
    print("TASK 5: BATCH EVALUATION ON 50 RANDOM FILES")
    print("="*60)
    print("This task uses the code from Task 3 to measure performance")
    print("on 50 randomly selected forum posts from the text directory.")
    print("="*60)
    
    cadec_root = './cadec'
    
    if not os.path.exists(cadec_root):
        print(f"Error: CADEC directory not found at {cadec_root}")
        return
    
    # Get 50 random files
    selected_files = get_50_random_files(cadec_root, seed=42)
    
    if not selected_files:
        print("No files found for evaluation!")
        return
    
    print(f"Sample of selected files: {selected_files[:5]}... (showing first 5)")
    
    # Initialize components
    print("\nInitializing Medical NER Tagger and Evaluator...")
    tagger = MedicalNERTagger()
    evaluator = CADECEvaluator(cadec_root)
    
    # Process all files
    print(f"\nProcessing {len(selected_files)} files using Task 3 methodology...")
    
    all_results = []
    total_time = 0
    successful_files = 0
    
    for i, filename in enumerate(selected_files):
        print(f"Processing {i+1}/{len(selected_files)}: {filename}")
        
        evaluation_results, processing_time, success = process_single_file_task5(
            filename, cadec_root, tagger, evaluator
        )
        
        total_time += processing_time
        
        if success and evaluation_results:
            all_results.append({
                'filename': filename,
                'metrics': evaluation_results,
                'processing_time': processing_time
            })
            successful_files += 1
            
            # Print brief progress
            if 'overall' in evaluation_results:
                overall = evaluation_results['overall']
                print(f"  F1: {overall['f1']:.3f}, P: {overall['precision']:.3f}, R: {overall['recall']:.3f}")
        else:
            print(f"  Failed to process")
        
        # Progress update every 10 files
        if (i + 1) % 10 == 0:
            avg_time = total_time / (i + 1)
            remaining_time = avg_time * (len(selected_files) - i - 1)
            print(f"  Progress: {i+1}/{len(selected_files)} | Success: {successful_files}/{i+1} | Est. remaining: {remaining_time:.1f}s")
    
    if not all_results:
        print("No successful evaluations completed!")
        return
    
    # Calculate comprehensive statistics
    print(f"\nCalculating comprehensive statistics...")
    stats = calculate_batch_statistics(all_results)
    
    # Print comprehensive summary
    print_comprehensive_summary(stats, len(all_results), total_time)
    
    # Save detailed results
    save_detailed_results(all_results)
    
    print(f"\nTask 5 completed successfully!")
    print(f"Successfully processed: {len(all_results)}/{len(selected_files)} files")
    print(f"Results saved in: ./task5_results/")

if __name__ == "__main__":
    main()