# Toxic Comment Detector
AI Toxic Comment Detection with Hugging Face

***
An AI-powered content moderation system that automatically identifies and classifies toxic, harmful, or offensive comments using state-of-the-art transformer models and ensemble learning techniques. 

**Live Demo:** [https://toxic-detector-4pic7qqpqwopsgx6prcr4r.streamlit.app/](https://toxic-detector-4pic7qqpqwopsgx6prcr4r.streamlit.app/)

## Overview

This application detects toxic content in online comments using an ensemble approach that combines multiple machine learning models to achieve high accuracy with reduced false positives. The system can classify comments into 6 distinct toxicity categories and provides confidence scores for transparent decision-making. 

### Real-World Applications
- **Social Media Platforms**: Automated moderation for Twitter, Facebook, Instagram comments
- **Gaming Communities**: Detecting toxic chat in multiplayer games
- **Online Forums**: Reddit, Stack Overflow community management
- **Customer Service**: Flagging abusive messages for priority handling

## Key Features

- **Multi-Label Classification**: Detects 6 toxicity categories simultaneously (toxic, severe_toxic, obscene, threat, insult, identity_hate) [ppl-ai-file-upload.s3.amazonaws]
- **Ensemble Learning**: Combines Detoxify (BERT-based) and Zero-Shot Classification models for 98.64% AUC accuracy 
- **Real-Time Analysis**: Processes single comments with detailed breakdowns in 1-2 seconds
- **Batch Processing**: Analyze multiple comments efficiently for bulk moderation tasks
- **Interactive Visualizations**: Token-level analysis and confidence score charts
- **Zero-Shot Capability**: Adaptable to new toxicity categories without retraining 

## Architecture

```
├── app.py              → Streamlit UI (Frontend)
├── toxic_detector.py   → ML Logic (Backend/Brain)
└── requirements.txt    → Dependencies
```

**Flow**: User Input → Streamlit UI → ToxicCommentDetector → ML Models → Ensemble Results → Display
## Technical Implementation

### Models Used

1. **Detoxify (Primary Detector)**
   - Pre-trained BERT model fine-tuned on 159,000 Wikipedia comments from Jigsaw Toxic Comment Challenge (2018)
   - Achieves 98.64% AUC score
   - Outputs 6 probability scores (0-1) for each toxicity category

2. **Facebook BART-Large-MNLI (Zero-Shot Classifier)**
   - Backup classifier requiring no training data for specific categories [ppl-ai-file-upload.s3.amazonaws]
   - Enables custom category definition on-the-fly
   - Provides alternative perspective to reduce false positives

3. **BERT Tokenizer**
   - Converts text into tokens for transformer processing [ppl-ai-file-upload.s3.amazonaws]
   - Visualizes how AI "sees" and interprets input text

### Core ML Concepts

- **Transformers (BERT Architecture)**: Uses attention mechanisms to understand context bidirectionally 
- **Multi-Label Classification**: Single comments can belong to multiple toxicity categories simultaneously 
- **Ensemble Methods**: Averages Detoxify and Zero-Shot scores for robust predictions 
- **Tokenization**: Breaks text into subword pieces for neural network processing

  
### Ensemble Decision Logic
```python
detoxify_score = detoxify_results['overall_toxicity']
zeroshot_score = zeroshot_results['confidence']
ensemble_score = (detoxify_score + zeroshot_score) / 2
is_toxic = ensemble_score > 0.5  # 50% threshold
```

## Toxicity Categories

| Category | Definition | Example |
|----------|------------|---------|
| **Toxic** | Rude, disrespectful, unreasonable | "This is garbage" |
| **Severe Toxic** | Very hateful, extremely aggressive | "I hope you die" |
| **Obscene** | Contains profanity/obscene language | "F*** you" |
| **Threat** | Threatens violence or harm | "I'll find you" |
| **Insult** | Direct personal attacks | "You're an idiot" |
| **Identity Hate** | Attacks based on race, religion, etc. | "All [group] are..." |


### Usage

**Single Comment Analysis:**
1. Navigate to "Single Analysis" tab
2. Enter comment text in the text area
3. Click "Analyze with LLM"
4. View detailed results with toxicity scores, primary category, and visualizations

**Batch Processing:**
1. Navigate to "Batch Analysis" tab
2. Enter multiple comments (one per line)
3. Click "Analyze Batch"
4. Download results as CSV for further analysis

## Analysis Features

### 1. Tokenization Visualization
Shows how transformers break down and interpret text at the token level: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61593299/4d0dd79f-6435-4919-88f2-2695eeaea8d1/llm-a2.pdf)
```
Input: "unhappiness"
Tokens: ['un', '##happiness']
Token IDs: [101, 1045, 5223, 2017, 102]
```

### 2. Text Feature Analysis
Examines patterns associated with toxic content: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61593299/4d0dd79f-6435-4919-88f2-2695eeaea8d1/llm-a2.pdf)
- Uppercase ratio (shouting indicator)
- Exclamation mark count (aggression)
- Caps lock words (anger)
- Character and word counts

### 3. Dual Model Scoring
Provides scores from both Detoxify and Zero-Shot models with ensemble average [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61593299/4d0dd79f-6435-4919-88f2-2695eeaea8d1/llm-a2.pdf)

### 4. Interactive Charts
- Bar charts showing confidence across all 6 categories
- Color-coded visualizations (red intensity = higher toxicity)

## Performance Metrics

- **Accuracy**: 98.64% AUC score on Jigsaw Challenge dataset [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61593299/4d0dd79f-6435-4919-88f2-2695eeaea8d1/llm-a2.pdf)
- **Processing Speed**: 1-2 seconds per comment average
- **Dataset**: Trained on 159,571 human-labeled Wikipedia comments [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61593299/4d0dd79f-6435-4919-88f2-2695eeaea8d1/llm-a2.pdf)

## Future Enhancements

- Multi-language support (currently English only)
- User feedback loop for continuous improvement
- API integration for real-time moderation pipelines
- Explainability features showing which words triggered detection
- Sarcasm detection specialized model
- GPU acceleration for production-scale deployment

## Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: PyTorch, Transformers (Hugging Face)
- **Models**: Detoxify, BART-Large-MNLI
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud

## Learning Resources

This project demonstrates:
- State-of-the-art NLP with transformers
- Production ML deployment practices
- Ensemble learning techniques
- Interactive web UI development
- Multi-label classification strategies

## Limitations

- Sarcasm detection not reliably supported [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61593299/4d0dd79f-6435-4919-88f2-2695eeaea8d1/llm-a2.pdf)
- English-language focused (pre-trained models)
- Requires human review for edge cases (recommendation: "FLAG FOR REVIEW" not auto-delete) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61593299/4d0dd79f-6435-4919-88f2-2695eeaea8d1/llm-a2.pdf)
- Cultural context understanding limited
