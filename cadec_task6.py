#!/usr/bin/env python3
"""
CADEC Dataset Analysis - Task 6: SNOMED Code Matching
For the same filename combine the information given in the sub-directories
original and sct to create a data structure that stores the information: standard
code, standard textual description of the code (as per SNOMED CT), label type,
ground truth text segment. Use this data structure to give the appropriate standard
code and standard text for each text segment that has the ADR label for the output
in Task 2. Do this in two different ways: a) using approximate string match and
b) using an embedding model from Hugging Face to match the two text segments.
"""

import os
import json
from difflib import SequenceMatcher
from collections import defaultdict

class SNOMEDCodeMatcher:
    def __init__(self, cadec_root):
        """
        Initialize SNOMED CT code matcher
        
        Args:
            cadec_root (str): Path to CADEC dataset root
        """
        self.cadec_root = cadec_root
        self.original_dir = os.path.join(cadec_root, 'original')
        self.sct_dir = os.path.join(cadec_root, 'sct')
        self.embedding_model = None
        
    def load_embedding_model(self):
        """
        Load sentence transformer model for semantic matching
        """
        if self.embedding_model is None:
            print("Loading sentence transformer model...")
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Loaded all-MiniLM-L6-v2 model for semantic matching")
                return True
            except Exception as e:
                print(f"⚠️  Could not load embedding model: {e}")
                print("String matching only will be used.")
                return False
        return True
    
    def parse_original_file(self, filepath):
        """
        Parse original annotation file
        Format: T1	ADR 9 19	bit drowsy
        
        Args:
            filepath (str): Path to original annotation file
            
        Returns:
            dict: Dictionary mapping tags to entity information
        """
        entities = {}
        
        if not os.path.exists(filepath):
            return entities
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
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
                        
                        entities[tag] = {
                            'label': label,
                            'ranges': ranges,
                            'text': text.strip()
                        }
        
        return entities
    
    def parse_sct_file(self, filepath):
        """
        Parse SCT (SNOMED CT) annotation file
        Format: TT1	271782001 | Drowsy | 9 19	bit drowsy
        
        Args:
            filepath (str): Path to SCT annotation file
            
        Returns:
            dict: Dictionary mapping tags to SNOMED CT information
        """
        sct_data = {}
        
        if not os.path.exists(filepath):
            return sct_data
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                # Parse SCT line format: TT1	271782001 | Drowsy | 9 19	bit drowsy
                parts = line.split('\t')
                if len(parts) >= 3:
                    tag = parts[0]  # TT1
                    code_desc_ranges = parts[1]  # "271782001 | Drowsy | 9 19"
                    text = parts[2]  # "bit drowsy"
                    
                    # Parse the code_desc_ranges part
                    codes_and_descriptions = []
                    
                    # Handle multiple codes (separated by "or")
                    code_sections = code_desc_ranges.split(' or ')
                    
                    for section in code_sections:
                        # Split by | to separate code, description, and ranges
                        if '|' in section:
                            pipe_parts = section.split('|')
                            if len(pipe_parts) >= 2:
                                # Extract code (first part before |)
                                code_part = pipe_parts[0].strip()
                                description = pipe_parts[1].strip()
                                
                                # The code might be followed by ranges, extract just the code
                                code_tokens = code_part.split()
                                code = code_tokens[0] if code_tokens else ""
                                
                                codes_and_descriptions.append({
                                    'code': code,
                                    'description': description
                                })
                        else:
                            # Fallback: try to parse without | separators
                            tokens = section.strip().split()
                            if tokens:
                                code = tokens[0]
                                description = ' '.join(tokens[1:]) if len(tokens) > 1 else ""
                                codes_and_descriptions.append({
                                    'code': code,
                                    'description': description
                                })
                    
                    sct_data[tag] = {
                        'codes_descriptions': codes_and_descriptions,
                        'text': text.strip()
                    }
        
        return sct_data
    
    def create_combined_dataset(self, filename):
        """
        Combine original and SCT data for a single file
        
        Args:
            filename (str): Filename without extension
            
        Returns:
            list: Combined dataset with SNOMED CT codes
        """
        print(f"Creating combined dataset for: {filename}")
        
        # Load original annotations
        original_file = os.path.join(self.original_dir, filename + '.ann')
        original_entities = self.parse_original_file(original_file)
        
        # Load SCT annotations  
        sct_file = os.path.join(self.sct_dir, filename + '.ann')
        sct_data = self.parse_sct_file(sct_file)
        
        # Combine data
        combined_data = []
        
        for tag, entity in original_entities.items():
            # Map to SCT tag (original T1 -> SCT TT1)
            sct_tag = 'T' + tag  # T1 -> TT1
            
            entity_data = {
                'tag': tag,
                'label': entity['label'],
                'text': entity['text'],
                'ranges': entity['ranges'],
                'snomed_codes': [],
                'snomed_descriptions': []
            }
            
            # Add SNOMED CT codes if available
            if sct_tag in sct_data:
                sct_entry = sct_data[sct_tag]
                for code_desc in sct_entry['codes_descriptions']:
                    entity_data['snomed_codes'].append(code_desc['code'])
                    entity_data['snomed_descriptions'].append(code_desc['description'])
            
            combined_data.append(entity_data)
        
        print(f"Combined {len(combined_data)} entities with SNOMED codes")
        return combined_data
    
    def load_predictions(self, filename, predictions_dir='./predictions'):
        """
        Load predictions from Task 2 for the given filename
        
        Args:
            filename (str): Filename without extension
            predictions_dir (str): Directory containing predictions
            
        Returns:
            list: List of predicted ADR entities
        """
        pred_file = os.path.join(predictions_dir, filename + '.ann')
        adr_predictions = []
        
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file not found - {pred_file}")
            return adr_predictions
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
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
                        
                        # Only keep ADR predictions
                        if label == 'ADR':
                            adr_predictions.append({
                                'tag': tag,
                                'label': label,
                                'ranges': ranges,
                                'text': text.strip()
                            })
        
        print(f"Found {len(adr_predictions)} ADR predictions")
        return adr_predictions
    
    def string_similarity(self, text1, text2):
        """
        Calculate string similarity using sequence matcher
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0-1)
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity
    
    def embedding_similarity(self, text1, text2):
        """
        Calculate semantic similarity using embeddings
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        if not self.load_embedding_model():
            return 0.0
        
        try:
            # Get embeddings
            embeddings = self.embedding_model.encode([text1, text2])
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating embedding similarity: {e}")
            return 0.0
    
    def match_adr_to_snomed_string(self, adr_predictions, combined_data, threshold=0.6):
        """
        Match ADR predictions to SNOMED codes using string similarity
        
        Args:
            adr_predictions (list): List of ADR predictions
            combined_data (list): Combined ground truth data with SNOMED codes
            threshold (float): Minimum similarity threshold
            
        Returns:
            list: List of matches with SNOMED codes
        """
        print(f"\nMethod A: String-based matching (threshold={threshold})")
        print("-" * 50)
        
        matches = []
        
        # Filter combined data to ADR entities only
        adr_ground_truth = [entity for entity in combined_data if entity['label'] == 'ADR']
        
        for pred in adr_predictions:
            pred_text = pred['text']
            best_match = None
            best_similarity = 0
            
            # Find best matching ground truth ADR
            for gt_entity in adr_ground_truth:
                for i, snomed_desc in enumerate(gt_entity['snomed_descriptions']):
                    # Calculate string similarity
                    similarity = self.string_similarity(pred_text, snomed_desc)
                    
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = {
                            'predicted_text': pred_text,
                            'predicted_tag': pred['tag'],
                            'predicted_ranges': pred['ranges'],
                            'matched_description': snomed_desc,
                            'snomed_code': gt_entity['snomed_codes'][i] if i < len(gt_entity['snomed_codes']) else 'N/A',
                            'ground_truth_text': gt_entity['text'],
                            'similarity': similarity,
                            'method': 'string'
                        }
            
            if best_match:
                matches.append(best_match)
                print(f"✅ '{pred_text}' -> SNOMED: {best_match['snomed_code']} '{best_match['matched_description']}' (sim: {best_similarity:.3f})")
            else:
                matches.append({
                    'predicted_text': pred_text,
                    'predicted_tag': pred['tag'],
                    'predicted_ranges': pred['ranges'],
                    'matched_description': None,
                    'snomed_code': None,
                    'ground_truth_text': None,
                    'similarity': 0.0,
                    'method': 'string'
                })
                print(f"❌ '{pred_text}' -> No match found")
        
        return matches
    
    def match_adr_to_snomed_embedding(self, adr_predictions, combined_data, threshold=0.7):
        """
        Match ADR predictions to SNOMED codes using embedding similarity
        
        Args:
            adr_predictions (list): List of ADR predictions
            combined_data (list): Combined ground truth data with SNOMED codes
            threshold (float): Minimum similarity threshold
            
        Returns:
            list: List of matches with SNOMED codes
        """
        print(f"\nMethod B: Embedding-based matching (threshold={threshold})")
        print("-" * 50)
        
        if not self.load_embedding_model():
            print("❌ Embedding model not available, skipping embedding matching")
            return []
        
        matches = []
        
        # Filter combined data to ADR entities only
        adr_ground_truth = [entity for entity in combined_data if entity['label'] == 'ADR']
        
        for pred in adr_predictions:
            pred_text = pred['text']
            best_match = None
            best_similarity = 0
            
            # Find best matching ground truth ADR
            for gt_entity in adr_ground_truth:
                for i, snomed_desc in enumerate(gt_entity['snomed_descriptions']):
                    # Calculate embedding similarity
                    similarity = self.embedding_similarity(pred_text, snomed_desc)
                    
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = {
                            'predicted_text': pred_text,
                            'predicted_tag': pred['tag'],
                            'predicted_ranges': pred['ranges'],
                            'matched_description': snomed_desc,
                            'snomed_code': gt_entity['snomed_codes'][i] if i < len(gt_entity['snomed_codes']) else 'N/A',
                            'ground_truth_text': gt_entity['text'],
                            'similarity': similarity,
                            'method': 'embedding'
                        }
            
            if best_match:
                matches.append(best_match)
                print(f"✅ '{pred_text}' -> SNOMED: {best_match['snomed_code']} '{best_match['matched_description']}' (sim: {best_similarity:.3f})")
            else:
                matches.append({
                    'predicted_text': pred_text,
                    'predicted_tag': pred['tag'],
                    'predicted_ranges': pred['ranges'],
                    'matched_description': None,
                    'snomed_code': None,
                    'ground_truth_text': None,
                    'similarity': 0.0,
                    'method': 'embedding'
                })
                print(f"❌ '{pred_text}' -> No match found")
        
        return matches
    
    def compare_methods(self, string_matches, embedding_matches):
        """
        Compare results from string and embedding matching methods
        
        Args:
            string_matches (list): String matching results
            embedding_matches (list): Embedding matching results
            
        Returns:
            dict: Comparison statistics
        """
        print(f"\n{'='*60}")
        print("COMPARISON OF MATCHING METHODS")
        print(f"{'='*60}")
        
        # Count successful matches
        string_success = sum(1 for match in string_matches if match['snomed_code'])
        embedding_success = sum(1 for match in embedding_matches if match['snomed_code'])
        total_predictions = len(string_matches)
        
        # Check agreement (when both methods find a match, do they agree?)
        agreement = 0
        both_matched = 0
        
        for str_match, emb_match in zip(string_matches, embedding_matches):
            str_found = str_match['snomed_code'] is not None
            emb_found = emb_match['snomed_code'] is not None
            
            if str_found and emb_found:
                both_matched += 1
                if str_match['snomed_code'] == emb_match['snomed_code']:
                    agreement += 1
        
        comparison = {
            'total_predictions': total_predictions,
            'string_matches': string_success,
            'embedding_matches': embedding_success,
            'both_matched': both_matched,
            'agreement': agreement,
            'string_only': string_success - both_matched if string_success >= both_matched else 0,
            'embedding_only': embedding_success - both_matched if embedding_success >= both_matched else 0
        }
        
        print(f"Total ADR predictions: {total_predictions}")
        print(f"String matching found: {string_success} codes ({string_success/total_predictions*100:.1f}%)")
        print(f"Embedding matching found: {embedding_success} codes ({embedding_success/total_predictions*100:.1f}%)")
        print(f"Both methods matched: {both_matched}")
        print(f"Agreement when both matched: {agreement}/{both_matched if both_matched > 0 else 1}")
        print(f"String-only matches: {comparison['string_only']}")
        print(f"Embedding-only matches: {comparison['embedding_only']}")
        
        return comparison
    
    def save_results(self, filename, string_matches, embedding_matches, comparison, output_dir='./task6_results'):
        """
        Save results to JSON file
        
        Args:
            filename (str): Base filename
            string_matches (list): String matching results
            embedding_matches (list): Embedding matching results
            comparison (dict): Comparison statistics
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'filename': filename,
            'string_matches': string_matches,
            'embedding_matches': embedding_matches,
            'comparison': comparison
        }
        
        output_file = os.path.join(output_dir, f'{filename}_snomed_matching.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")

def main():
    """
    Main function for Task 6
    """
    print("="*60)
    print("TASK 6: SNOMED CODE MATCHING WITH STRING AND EMBEDDING METHODS")
    print("="*60)
    print("Combining original and SCT data to match ADR predictions to SNOMED codes")
    print("="*60)
    
    cadec_root = './cadec'
    
    if not os.path.exists(cadec_root):
        print(f"Error: CADEC directory not found at {cadec_root}")
        return
    
    # Initialize matcher
    matcher = SNOMEDCodeMatcher(cadec_root)
    
    # Use the first available file with predictions
    predictions_dir = './predictions'
    if not os.path.exists(predictions_dir):
        print(f"Error: Predictions directory not found at {predictions_dir}")
        print("Please run Task 2 first to generate predictions.")
        return
    
    # Get first prediction file
    pred_files = [f for f in os.listdir(predictions_dir) if f.endswith('.ann')]
    if not pred_files:
        print("No prediction files found!")
        return
    
    filename = pred_files[0].replace('.ann', '')
    print(f"Processing file: {filename}")
    
    # Step 1: Create combined dataset
    combined_data = matcher.create_combined_dataset(filename)
    
    if not combined_data:
        print("No combined data found!")
        return
    
    # Step 2: Load ADR predictions from Task 2
    adr_predictions = matcher.load_predictions(filename, predictions_dir)
    
    if not adr_predictions:
        print("No ADR predictions found!")
        return
    
    # Step 3: Match using string similarity
    string_matches = matcher.match_adr_to_snomed_string(adr_predictions, combined_data)
    
    # Step 4: Match using embedding similarity
    embedding_matches = matcher.match_adr_to_snomed_embedding(adr_predictions, combined_data)
    
    # Step 5: Compare methods
    comparison = matcher.compare_methods(string_matches, embedding_matches)
    
    # Step 6: Save results
    matcher.save_results(filename, string_matches, embedding_matches, comparison)
    
    print(f"\nTask 6 completed!")
    print(f"✅ Created combined dataset with SNOMED codes")
    print(f"✅ Matched ADR predictions using string similarity")
    print(f"✅ Matched ADR predictions using embedding similarity")
    print(f"✅ Compared both methods")
    print(f"✅ Results saved to ./task6_results/")

if __name__ == "__main__":
    main()