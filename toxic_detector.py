"""
Toxic Comment Detection using Hugging Face Transformers
Shows LLM Concepts: NER, Token Classification, Embeddings
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from detoxify import Detoxify
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class ToxicCommentDetector:
    """
    Multi-model toxic comment detection system
    Shows: Transformer architecture, Tokenization, Multi-label classification
    """
    
    def __init__(self):
        """Initialize multiple detection models"""
        print("🔄 Loading models...")
        
        # Load Detoxify model (trained on Jigsaw dataset - 159K comments)
        # Shows: Pre-trained transformer fine-tuned for toxicity
        self.detoxify_model = Detoxify('original')
        
        # Load zero-shot classifier for custom toxic categories
        # Shows: Zero-shot learning (no training needed)
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Load transformer for tokenization analysis
        # Shows: Tokenization and attention mechanism
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Toxicity categories
        self.toxicity_types = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate"
        ]
        
        print("✅ Models loaded successfully!")
    
    def tokenize_and_visualize(self, text: str) -> Dict:
        """
        Tokenize text and show how transformer sees it
        Shows: Tokenization, Token IDs, Embeddings concept
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text, return_tensors='pt')
        
        # Get attention mask
        attention_mask = torch.ones_like(token_ids)
        
        return {
            'text': text,
            'tokens': tokens,
            'token_ids': token_ids.tolist(),
            'attention_mask': attention_mask.tolist(),
            'num_tokens': len(tokens),
            'max_length': len(token_ids[0].tolist())
        }
    
    def detect_toxicity_detoxify(self, text: str) -> Dict:
        """
        Detect toxicity using Detoxify model (unitary/toxic-bert)
        Shows: Transformer-based classification, Multi-label prediction
        
        Model: BERT fine-tuned on 159K Wikipedia comments + Civil Comments
        Trained on Jigsaw challenges with 98.64% AUC score
        """
        # Predict toxicity
        results = self.detoxify_model.predict(text)
        
        # Format results
        toxicity_scores = {
            'toxic': float(results['toxicity']),
            'severe_toxic': float(results.get('severe_toxicity', 0)),
            'obscene': float(results.get('obscene', 0)),
            'threat': float(results.get('threat', 0)),
            'insult': float(results.get('insult', 0)),
            'identity_hate': float(results.get('identity_hate', 0))
        }
        
        # Determine overall toxicity
        overall_toxicity = max(toxicity_scores.values())
        is_toxic = overall_toxicity > 0.5
        
        return {
            'text': text,
            'overall_toxicity': overall_toxicity,
            'is_toxic': is_toxic,
            'scores': toxicity_scores,
            'model': 'detoxify (BERT)',
            'accuracy': '98.64% (Jigsaw Challenge)',
            'training_data': '159K Wikipedia comments'
        }
    
    def categorize_toxicity_zeroshot(self, text: str, custom_categories: List[str] = None) -> Dict:
        """
        Categorize toxicity using Zero-Shot Classification
        Shows: Zero-shot learning (prompt engineering concept)
        No fine-tuning needed - defines categories on-the-fly
        """
        if custom_categories is None:
            custom_categories = self.toxicity_types
        
        # Zero-shot classification
        result = self.zero_shot_classifier(
            text,
            custom_categories,
            multi_class=True
        )
        
        return {
            'text': text,
            'predicted_categories': result['labels'],
            'scores': dict(zip(result['labels'], result['scores'])),
            'primary_category': result['labels'][0],
            'confidence': result['scores'][0],
            'method': 'Zero-Shot Classification',
            'model': 'facebook/bart-large-mnli'
        }
    
    def analyze_text_features(self, text: str) -> Dict:
        """
        Extract text features that indicate toxicity
        Shows: Feature engineering, Pattern recognition
        """
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'exclamation_marks': text.count('!'),
            'question_marks': text.count('?'),
            'special_chars': sum(1 for c in text if not c.isalnum() and c != ' '),
            'profanity_indicators': self._check_profanity_patterns(text),
            'caps_lock_count': len([w for w in text.split() if w.isupper() and len(w) > 1])
        }
        return features
    
    def _check_profanity_patterns(self, text: str) -> List[str]:
        """Check for profanity-like patterns"""
        profanity_patterns = ['***', '###', '@@', '!!!']
        found = [p for p in profanity_patterns if p in text]
        return found if found else []
    
    def full_analysis(self, text: str) -> Dict:
        """
        Complete toxicity analysis combining multiple methods
        Shows: Ensemble approach, Multi-model prediction
        """
        print(f"\n🔍 Analyzing: '{text[:50]}...'")
        
        analysis = {
            'input_text': text,
            'tokenization': self.tokenize_and_visualize(text),
            'detoxify_results': self.detect_toxicity_detoxify(text),
            'zeroshot_results': self.categorize_toxicity_zeroshot(text),
            'text_features': self.analyze_text_features(text),
            'final_verdict': None
        }
        
        # Determine final verdict
        detoxify_score = analysis['detoxify_results']['overall_toxicity']
        zeroshot_score = analysis['zeroshot_results']['confidence']
        
        # Ensemble decision
        ensemble_score = (detoxify_score + zeroshot_score) / 2
        
        analysis['final_verdict'] = {
            'ensemble_toxicity_score': ensemble_score,
            'is_toxic': ensemble_score > 0.5,
            'confidence': max(detoxify_score, zeroshot_score),
            'recommendation': 'FLAG FOR REVIEW' if ensemble_score > 0.5 else 'APPROVE',
            'reasoning': f"Detoxify: {detoxify_score:.2%}, Zero-Shot: {zeroshot_score:.2%}"
        }
        
        return analysis
    
    def batch_analyze(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze multiple texts at once
        """
        results = []
        for text in texts:
            analysis = self.full_analysis(text)
            results.append({
                'text': text,
                'toxicity_score': analysis['detoxify_results']['overall_toxicity'],
                'is_toxic': analysis['detoxify_results']['is_toxic'],
                'primary_category': analysis['zeroshot_results']['primary_category'],
                'verdict': analysis['final_verdict']['recommendation']
            })
        
        return pd.DataFrame(results)

