#!/usr/bin/env python3
"""
CADEC Dataset Analysis - Task 4: ADR-specific Evaluation using MedDRA
Repeat the performance calculation in Task 3 but now only for the label type ADR
where the ground truth is now chosen from the sub-directory meddra.
"""

import os
import random
from cadec_task2 import MedicalNERTagger
from cadec_task3 import CADECEvaluator
import time

def get_files_with_adr_annotations(cadec_root, n=5, seed=42):
    """
    Get files that actually have ADR annotations in MedDRA
    
    Args:
        cadec_root (str): CADEC root directory
        n (int): Number of files to return
        seed (int): Random seed
        
    Returns:
        list: Filenames with ADR annotations
    """
    meddra_dir = os.path.join(cadec_root, 'meddra')
    
    if not os.path.exists(meddra_dir):
        print(f"Error: MedDRA directory not found at {meddra_dir}")
        return []
    
    files_with_adr = []
    
    # Check all MedDRA files for ADR content
    for filename in os.listdir(meddra_dir):
        if filename.endswith('.ann'):
            filepath = os.path.join(meddra_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Check if file has actual content (not just comments or empty)
                    if content and not all(line.startswith('#') or line.strip() == '' for line in content.split('\n')):
                        files_with_adr.append(filename.replace('.ann', ''))
            except Exception:
                continue
    
    print(f"Found {len(files_with_adr)} files with MedDRA ADR annotations")
    
    # Randomly sample
    random.seed(seed)
    selected = random.sample(files_with_adr, min(n, len(files_with_adr)))
    
    return selected

def evaluate_single_file_adr_meddra(filename, cadec_root, tagger, evaluator):
    """
    Evaluate a single file for ADR performance using MedDRA ground truth
    
    Args:
        filename (str): Filename without extension
        cadec_root (str): CADEC root directory
        tagger: Medical NER tagger instance
        evaluator: CADEC evaluator instance
        
    Returns:
        tuple: (adr_metrics, processing_time, success)
    """
    start_time = time.time()
    
    # Generate predictions
    text_filepath = os.path.join(cadec_root, 'text', filename + '.txt')
    
    try:
        if not os.path.exists(text_filepath):
            print(f"Warning: Text file not found - {text_filepath}")
            return None, time.time() - start_time, False
        
        # Process text file
        original_text, bio_tagged_words, annotations = tagger.process_text_file(text_filepath)
        
        # Save predictions
        predictions_dir = './task4_predictions'
        os.makedirs(predictions_dir, exist_ok=True)
        pred_filepath = os.path.join(predictions_dir, filename + '.ann')
        tagger.save_annotations(annotations, pred_filepath)
        
        # Evaluate ONLY ADR using MedDRA ground truth
        results = evaluator.evaluate_single_file(filename, predictions_dir, use_meddra=True)
        
        processing_time = time.time() - start_time
        
        # Extract ADR-specific metrics
        if 'ADR' in results:
            adr_metrics = results['ADR']
            return adr_metrics, processing_time, True
        else:
            # No ADR entities found
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}, processing_time, True
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, time.time() - start_time, False

def print_adr_results(filename, adr_metrics, processing_time):
    """
    Print ADR evaluation results for a single file
    
    Args:
        filename (str): Filename
        adr_metrics (dict): ADR metrics
        processing_time (float): Processing time
    """
    print(f"\n{'='*60}")
    print(f"ADR EVALUATION RESULTS - {filename}")
    print(f"{'='*60}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Ground truth source: MedDRA annotations")
    print(f"")
    print(f"ADR PERFORMANCE:")
    print(f"  Precision: {adr_metrics['precision']:.3f}")
    print(f"  Recall:    {adr_metrics['recall']:.3f}")
    print(f"  F1-Score:  {adr_metrics['f1']:.3f}")
    print(f"  Support:   {adr_metrics['support']} (total ADR entities in ground truth)")
    print(f"{'='*60}")

def calculate_overall_adr_performance(all_adr_results):
    """
    Calculate overall ADR performance across all files
    
    Args:
        all_adr_results (list): List of ADR metrics for each file
        
    Returns:
        dict: Overall ADR performance metrics
    """
    if not all_adr_results:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'total_support': 0}
    
    # Calculate micro-averaged metrics (aggregate TP, FP, FN across all files)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_support = 0
    
    for metrics in all_adr_results:
        support = metrics['support']
        recall = metrics['recall']
        precision = metrics['precision']
        
        # Calculate TP, FP, FN from precision, recall, support
        if support > 0:
            tp = int(recall * support)
            if precision > 0:
                predicted_positive = tp / precision
                fp = int(predicted_positive - tp)
            else:
                fp = 0
            fn = support - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_support += support
    
    # Calculate overall metrics
    if total_tp + total_fp > 0:
        overall_precision = total_tp / (total_tp + total_fp)
    else:
        overall_precision = 0.0
    
    if total_tp + total_fn > 0:
        overall_recall = total_tp / (total_tp + total_fn)
    else:
        overall_recall = 0.0
    
    if overall_precision + overall_recall > 0:
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    else:
        overall_f1 = 0.0
    
    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'total_support': total_support,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }

def main():
    """
    Main function for Task 4
    """
    print("="*60)
    print("TASK 4: ADR-SPECIFIC EVALUATION USING MEDDRA")
    print("="*60)
    print("This task repeats the performance calculation from Task 3")
    print("but focuses ONLY on ADR labels using MedDRA ground truth")
    print("instead of original annotations.")
    print("="*60)
    
    cadec_root = './cadec'
    
    if not os.path.exists(cadec_root):
        print(f"Error: CADEC directory not found at {cadec_root}")
        return
    
    # Initialize evaluator and tagger
    print("Initializing components...")
    evaluator = CADECEvaluator(cadec_root)
    tagger = MedicalNERTagger()
    
    # Get sample files that have MedDRA ADR annotations
    sample_files = get_files_with_adr_annotations(cadec_root, n=5, seed=42)
    
    if not sample_files:
        print("No files with MedDRA ADR annotations found!")
        return
    
    print(f"Selected files with MedDRA ADR annotations: {sample_files}")
    
    # Process each file
    all_adr_results = []
    total_processing_time = 0
    successful_files = 0
    
    for i, filename in enumerate(sample_files):
        print(f"\nProcessing {i+1}/{len(sample_files)}: {filename}")
        
        adr_metrics, processing_time, success = evaluate_single_file_adr_meddra(
            filename, cadec_root, tagger, evaluator
        )
        
        total_processing_time += processing_time
        
        if success and adr_metrics:
            print_adr_results(filename, adr_metrics, processing_time)
            all_adr_results.append(adr_metrics)
            successful_files += 1
        else:
            print(f"Failed to process {filename}")
    
    # Calculate and print overall results
    if all_adr_results:
        overall_performance = calculate_overall_adr_performance(all_adr_results)
        
        print(f"\n{'='*70}")
        print(f"TASK 4 OVERALL RESULTS: ADR EVALUATION WITH MEDDRA")
        print(f"{'='*70}")
        print(f"Files successfully processed: {successful_files}/{len(sample_files)}")
        print(f"Total processing time: {total_processing_time:.1f} seconds")
        print(f"Average time per file: {total_processing_time / len(sample_files):.2f} seconds")
        print(f"")
        print(f"METHODOLOGY:")
        print(f"• Same overlap-based evaluation as Task 3 (50% IoU threshold)")
        print(f"• Ground truth: MedDRA annotations (./cadec/meddra/)")
        print(f"• Focus: ONLY ADR label type")
        print(f"• Evaluation: Precision, Recall, F1-score for ADR detection")
        print(f"")
        print(f"OVERALL ADR PERFORMANCE:")
        print(f"  True Positives:  {overall_performance['total_tp']}")
        print(f"  False Positives: {overall_performance['total_fp']}")
        print(f"  False Negatives: {overall_performance['total_fn']}")
        print(f"  Precision:       {overall_performance['precision']:.3f}")
        print(f"  Recall:          {overall_performance['recall']:.3f}")
        print(f"  F1-Score:        {overall_performance['f1']:.3f}")
        print(f"  Total ADR Support: {overall_performance['total_support']}")
        print(f"")
        
        # Performance interpretation
        f1_score = overall_performance['f1']
        if f1_score > 0.7:
            performance_level = "EXCELLENT"
        elif f1_score > 0.5:
            performance_level = "GOOD"
        elif f1_score > 0.3:
            performance_level = "MODERATE"
        else:
            performance_level = "NEEDS IMPROVEMENT"
        
        print(f"PERFORMANCE LEVEL: {performance_level} (F1 = {f1_score:.3f})")
        print(f"")
        print(f"KEY DIFFERENCE FROM TASK 3:")
        print(f"• Task 3: Used original/ annotations as ground truth")
        print(f"• Task 4: Uses meddra/ annotations as ground truth")
        print(f"• Task 4: Focuses only on ADR entities (not Drug/Disease/Symptom)")
        print(f"{'='*70}")
    else:
        print("\nNo successful evaluations completed!")

if __name__ == "__main__":
    main()