#!/usr/bin/env python3
"""
CADEC Dataset Analysis - Task 2: LLM-based BIO Tagging (Final Updated)
Use Hugging Face models to label medical text with BIO format
Works with actual CADEC file format: T1	ADR 9 19	bit drowsy
"""

import os
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

class MedicalNERTagger:
    def __init__(self, model_name="Clinical-AI-Apollo/Medical-NER"):
        """
        Initialize the medical NER tagger
        
        Args:
            model_name (str): Hugging Face model name for medical NER
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Try different pre-trained medical NER models
        medical_ner_models = [
            "Clinical-AI-Apollo/Medical-NER",
            "d4data/biomedical-ner-all", 
            "samrawal/bert-base-uncased_clinical-ner",
            "emilyalsentzer/Bio_ClinicalBERT"
        ]
        
        self.ner_pipeline = None
        
        for model in medical_ner_models:
            try:
                print(f"Trying to load: {model}")
                self.ner_pipeline = pipeline(
                    "ner",
                    model=model,
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                print(f"✅ Successfully loaded {model}")
                self.model_name = model
                break
            except Exception as e:
                print(f"Failed to load {model}: {e}")
                continue
        
        # If all medical models fail, use general NER as fallback
        if self.ner_pipeline is None:
            print("All medical models failed, using general NER model...")
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
                print("✅ Fallback model loaded")
            except Exception as e:
                print(f"Even fallback model failed: {e}")
                raise e
    
    def map_entity_label(self, original_label):
        """
        Map model's entity labels to our target labels (ADR, Drug, Disease, Symptom)
        
        Args:
            original_label (str): Original entity label from the model
            
        Returns:
            str: Mapped label or None
        """
        # Normalize label
        normalized_label = original_label.upper().replace('_', '').replace('-', '')
        
        # Direct mapping
        if normalized_label in ['ADR', 'DRUG', 'DISEASE', 'SYMPTOM']:
            return normalized_label.capitalize()
        
        # Enhanced label mapping for biomedical NER models
        label_mapping = {
            # Drug related
            'CHEMICAL': 'Drug',
            'DRUG': 'Drug',
            'MEDICATION': 'Drug',
            'MEDICINE': 'Drug',
            'PHARMACEUTICAL': 'Drug',
            'TREATMENT': 'Drug',
            
            # Disease related
            'DISEASE': 'Disease',
            'DISORDER': 'Disease', 
            'CONDITION': 'Disease',
            'ILLNESS': 'Disease',
            'PATHOLOGY': 'Disease',
            
            # Symptom related
            'SYMPTOM': 'Symptom',
            'SIGN': 'Symptom',
            'FINDING': 'Symptom',
            'MANIFESTATION': 'Symptom',
            
            # ADR related (side effects)
            'ADVERSE': 'ADR',
            'SIDEEFFECT': 'ADR',
            'REACTION': 'ADR',
            'EFFECT': 'ADR'
        }
        
        # Check mapping dictionary
        for key, value in label_mapping.items():
            if key in normalized_label:
                return value
        
        # Heuristic mapping for general NER models
        if any(term in normalized_label for term in ['CHEM', 'MED', 'PHARM', 'TREAT']):
            return 'Drug'
        elif any(term in normalized_label for term in ['DIS', 'ILL', 'PATH', 'CONDITION']):
            return 'Disease'
        elif any(term in normalized_label for term in ['SYM', 'SIGN', 'PAIN', 'ACHE']):
            return 'Symptom'
        elif any(term in normalized_label for term in ['ADVERSE', 'SIDE', 'REACTION']):
            return 'ADR'
        
        # For PERSON/ORG entities, skip them
        if any(term in normalized_label for term in ['PERSON', 'ORG', 'LOC', 'MISC']):
            return None
        
        # Default: if it's a recognized entity but unknown type, 
        # apply medical context rules
        return self.apply_medical_context_rules(normalized_label)
    
    def apply_medical_context_rules(self, normalized_label):
        """
        Apply medical context rules for ambiguous entities
        
        Args:
            normalized_label (str): Normalized entity label
            
        Returns:
            str: Mapped label or None
        """
        # If it's any kind of entity but we're not sure,
        # we'll let the text content help us decide in extract_entities_with_pipeline
        return 'Symptom'  # Conservative default
    
    def extract_entities_with_pipeline(self, text):
        """
        Extract entities using the Hugging Face NER pipeline with medical context
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of entity dictionaries
        """
        try:
            # First try the NER pipeline
            entities = self.ner_pipeline(text)
            
            # If the pipeline gives poor results, use rule-based approach
            if len(entities) == 0 or self.are_results_poor(entities, text):
                print("Pipeline results poor, using rule-based medical NER...")
                return self.rule_based_medical_ner(text)
            
            # Post-process to map to our categories
            processed_entities = []
            for entity in entities:
                # Clean entity text
                entity_text = entity['word'].replace('##', '').strip()
                
                # Map entity labels to our target labels
                mapped_label = self.map_entity_label(entity['entity_group'])
                
                if mapped_label:
                    # Apply medical context refinement
                    refined_label = self.refine_label_with_context(entity_text, mapped_label, text)
                    
                    processed_entities.append({
                        'text': entity_text,
                        'label': refined_label,
                        'start': entity['start'],
                        'end': entity['end'],
                        'confidence': entity['score']
                    })
            
            return processed_entities
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            print("Falling back to rule-based medical NER...")
            return self.rule_based_medical_ner(text)
    
    def are_results_poor(self, entities, text):
        """
        Check if NER results are poor quality
        
        Args:
            entities (list): NER results
            text (str): Original text
            
        Returns:
            bool: True if results seem poor
        """
        if len(entities) == 0:
            return True
        
        # Check if too many entities (likely over-tagging)
        if len(entities) > len(text.split()) * 0.8:
            return True
        
        # Check if entities are mostly single characters or function words
        poor_entities = 0
        for entity in entities:
            word = entity['word'].replace('##', '').strip()
            if len(word) <= 2 or word.lower() in ['i', 'a', 'the', 'and', 'or', 'but', 'so', 'to', 'of', 'in', 'on']:
                poor_entities += 1
        
        if poor_entities > len(entities) * 0.5:
            return True
        
        return False
    
    def rule_based_medical_ner(self, text):
        """
        Rule-based medical NER as fallback
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of entity dictionaries
        """
        import re
        
        entities = []
        
        # Define medical dictionaries
        drugs = [
            'arthrotec', 'lipitor', 'voltaren', 'diclofenac', 'aspirin', 'ibuprofen',
            'naproxen', 'medication', 'medicine', 'pill', 'tablet', 'capsule'
        ]
        
        adrs = [
            'drowsy', 'drowsiness', 'blurred vision', 'nausea', 'headache', 'dizziness',
            'stomach irritation', 'gastric problems', 'side effect', 'weird', 'tired'
        ]
        
        diseases = [
            'arthritis', 'diabetes', 'hypertension', 'cancer', 'infection', 'disease',
            'disorder', 'condition'
        ]
        
        symptoms = [
            'pain', 'pains', 'ache', 'agony', 'discomfort', 'swelling', 'fever',
            'tears', 'hurt', 'sore'
        ]
        
        # Find entities using regex
        for drug in drugs:
            for match in re.finditer(r'\b' + re.escape(drug) + r'\b', text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'label': 'Drug',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        for adr in adrs:
            for match in re.finditer(r'\b' + re.escape(adr) + r'\b', text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'label': 'ADR',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        for disease in diseases:
            for match in re.finditer(r'\b' + re.escape(disease) + r'\b', text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'label': 'Disease',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        for symptom in symptoms:
            for match in re.finditer(r'\b' + re.escape(symptom) + r'\b', text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'label': 'Symptom',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        # Sort by start position and remove overlaps
        entities.sort(key=lambda x: x['start'])
        
        # Remove overlapping entities (keep longer ones)
        filtered_entities = []
        for entity in entities:
            overlap = False
            for existing in filtered_entities:
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    overlap = True
                    break
            if not overlap:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def refine_label_with_context(self, entity_text, initial_label, full_text):
        """
        Refine entity label based on medical context and text content
        
        Args:
            entity_text (str): The entity text
            initial_label (str): Initial label from mapping
            full_text (str): Full text context
            
        Returns:
            str: Refined label
        """
        entity_lower = entity_text.lower()
        
        # Known drug names/patterns
        drug_patterns = [
            'arthrotec', 'lipitor', 'voltaren', 'diclofenac', 'aspirin',
            'medication', 'pill', 'tablet', 'capsule', 'mg', 'dose'
        ]
        
        # Known ADR/side effect patterns  
        adr_patterns = [
            'drowsy', 'drowsiness', 'blurred vision', 'nausea', 'headache',
            'dizziness', 'stomach irritation', 'side effect', 'reaction'
        ]
        
        # Known disease patterns
        disease_patterns = [
            'arthritis', 'diabetes', 'hypertension', 'cancer', 'infection'
        ]
        
        # Known symptom patterns
        symptom_patterns = [
            'pain', 'ache', 'agony', 'discomfort', 'swelling', 'fever'
        ]
        
        # Check for drug patterns
        if any(pattern in entity_lower for pattern in drug_patterns):
            return 'Drug'
        
        # Check for ADR patterns
        if any(pattern in entity_lower for pattern in adr_patterns):
            return 'ADR'
        
        # Check for disease patterns  
        if any(pattern in entity_lower for pattern in disease_patterns):
            return 'Disease'
        
        # Check for symptom patterns
        if any(pattern in entity_lower for pattern in symptom_patterns):
            return 'Symptom'
        
        # Context-based refinement
        # If the entity appears after "taking", "on", it's likely a drug
        if any(phrase in full_text.lower() for phrase in [f'taking {entity_lower}', f'on {entity_lower}']):
            return 'Drug'
        
        # If entity appears with "feel", "have", it's likely ADR/symptom
        if any(phrase in full_text.lower() for phrase in [f'feel {entity_lower}', f'have {entity_lower}']):
            return 'ADR'
        
        return initial_label
    
    def text_to_bio_tags(self, text):
        """
        Convert text to BIO format tags
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of (word, BIO_tag) tuples
        """
        # Split text into words (simple whitespace tokenization)
        words = text.split()
        
        # Extract entities
        entities = self.extract_entities_with_pipeline(text)
        
        # Initialize all words as 'O' (Outside)
        bio_tags = ['O'] * len(words)
        
        # Convert character-based entity positions to word-based positions
        word_positions = []
        char_pos = 0
        for i, word in enumerate(words):
            start_pos = text.find(word, char_pos)
            if start_pos == -1:  # Word not found, estimate position
                start_pos = char_pos
            end_pos = start_pos + len(word)
            word_positions.append((start_pos, end_pos))
            char_pos = end_pos
        
        # Assign BIO tags
        for entity in entities:
            entity_start = entity['start']
            entity_end = entity['end']
            label = entity['label']
            
            # Find overlapping words
            first_word = True
            for i, (word_start, word_end) in enumerate(word_positions):
                # Check if word overlaps with entity (with some tolerance)
                if (word_start < entity_end and word_end > entity_start):
                    if first_word:
                        bio_tags[i] = f'B-{label}'
                        first_word = False
                    else:
                        bio_tags[i] = f'I-{label}'
        
        return list(zip(words, bio_tags))
    
    def bio_to_original_format(self, bio_tagged_words, original_text):
        """
        Convert BIO format to original annotation format
        
        Args:
            bio_tagged_words (list): List of (word, BIO_tag) tuples
            original_text (str): Original text for position calculation
            
        Returns:
            list: List of annotation tuples (tag, label, ranges, text)
        """
        annotations = []
        current_entity = None
        current_words = []
        tag_counter = 1
        
        for word, bio_tag in bio_tagged_words:
            if bio_tag.startswith('B-'):
                # Save previous entity if exists
                if current_entity:
                    annotations.append(self.create_annotation(
                        current_entity, current_words, original_text, tag_counter
                    ))
                    tag_counter += 1
                
                # Start new entity
                current_entity = bio_tag[2:]  # Remove 'B-'
                current_words = [word]
                
            elif bio_tag.startswith('I-') and current_entity:
                # Continue current entity
                current_words.append(word)
                
            else:  # 'O' tag
                # Save current entity if exists
                if current_entity:
                    annotations.append(self.create_annotation(
                        current_entity, current_words, original_text, tag_counter
                    ))
                    tag_counter += 1
                    current_entity = None
                    current_words = []
        
        # Save final entity if exists
        if current_entity:
            annotations.append(self.create_annotation(
                current_entity, current_words, original_text, tag_counter
            ))
        
        return annotations
    
    def create_annotation(self, label, words, original_text, tag_id):
        """
        Create annotation tuple in original format
        
        Args:
            label (str): Entity label
            words (list): List of words in the entity
            original_text (str): Original text
            tag_id (int): Tag identifier
            
        Returns:
            tuple: (tag, label, ranges, text)
        """
        entity_text = ' '.join(words)
        
        # Find position in original text (case insensitive)
        start_pos = original_text.lower().find(entity_text.lower())
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            ranges = f"{start_pos} {end_pos}"
            # Get the actual text from original (preserving case)
            actual_text = original_text[start_pos:end_pos]
        else:
            # Fallback: try individual words
            start_pos = original_text.lower().find(words[0].lower())
            if start_pos != -1:
                end_pos = start_pos + len(words[0])
                ranges = f"{start_pos} {end_pos}"
                actual_text = original_text[start_pos:end_pos]
            else:
                ranges = "0 0"
                actual_text = entity_text
        
        tag = f"T{tag_id}"
        
        return (tag, label, ranges, actual_text)
    
    def process_text_file(self, text_filepath):
        """
        Process a single text file and return annotations
        
        Args:
            text_filepath (str): Path to text file
            
        Returns:
            tuple: (original_text, bio_tagged_words, annotations)
        """
        # Read the text file
        with open(text_filepath, 'r', encoding='utf-8') as f:
            original_text = f.read().strip()
        
        print(f"Processing text: {original_text[:100]}...")
        
        # Convert to BIO format
        bio_tagged_words = self.text_to_bio_tags(original_text)
        
        # Convert to original annotation format
        annotations = self.bio_to_original_format(bio_tagged_words, original_text)
        
        return original_text, bio_tagged_words, annotations
    
    def save_annotations(self, annotations, output_filepath):
        """
        Save annotations to file in original format
        
        Args:
            annotations (list): List of annotation tuples
            output_filepath (str): Output file path
        """
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for tag, label, ranges, text in annotations:
                # Format: T1	ADR 9 19	bit drowsy (tab between tag and "label ranges", tab before text)
                f.write(f"{tag}\t{label} {ranges}\t{text}\n")
    
    def get_first_text_file(self, cadec_root):
        """Get the first available text file"""
        text_dir = os.path.join(cadec_root, 'text')
        if not os.path.exists(text_dir):
            return None
        
        text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
        return sorted(text_files)[0] if text_files else None

def demo_single_file(cadec_root):
    """
    Demonstrate the tagging process on the first available file
    
    Args:
        cadec_root (str): Path to CADEC dataset root
    """
    
    # Initialize tagger
    tagger = MedicalNERTagger()
    
    # Get first available text file
    filename = tagger.get_first_text_file(cadec_root)
    if not filename:
        print("No text files found!")
        return
    
    print(f"Processing file: {filename}")
    
    # Process the file
    text_filepath = os.path.join(cadec_root, 'text', filename)
    
    original_text, bio_tagged_words, annotations = tagger.process_text_file(text_filepath)
    
    print(f"\nOriginal text preview:")
    print(original_text[:200] + "..." if len(original_text) > 200 else original_text)
    
    print(f"\nBIO tagged words (first 20):")
    for word, tag in bio_tagged_words[:20]:
        print(f"{word:15} -> {tag}")
    
    print(f"\nGenerated annotations:")
    for annotation in annotations:
        print(f"{annotation[0]}\t{annotation[1]} {annotation[2]}\t{annotation[3]}")
    
    # Save results
    output_dir = "./predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = filename.replace('.txt', '')
    bio_output = os.path.join(output_dir, base_name + '_bio.txt')
    with open(bio_output, 'w', encoding='utf-8') as f:
        for word, tag in bio_tagged_words:
            f.write(f"{word}\t{tag}\n")
    
    ann_output = os.path.join(output_dir, base_name + '.ann')
    tagger.save_annotations(annotations, ann_output)
    
    print(f"\nResults saved to:")
    print(f"BIO format: {bio_output}")
    print(f"Annotations: {ann_output}")

def main():
    """Main function for Task 2"""
    cadec_root = './cadec'
    
    if not os.path.exists(cadec_root):
        print(f"Error: CADEC directory not found at {cadec_root}")
        print("Please download and extract CADEC.v2.zip first.")
        return
    
    # Demo on the first available file
    demo_single_file(cadec_root)

if __name__ == "__main__":
    main()