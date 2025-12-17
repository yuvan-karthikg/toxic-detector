"""
Toxic Comment Detection Web App
Streamlit UI for Toxic Comment Detection
Based on: GitHub projects + Hugging Face transformers
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from toxic_detector import ToxicCommentDetector
import time

# Page config
st.set_page_config(
    page_title="🚨 Toxic Comment Detector",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .toxic-badge {
        background-color: #FF6B6B;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .safe-badge {
        background-color: #51CF66;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .info-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None

@st.cache_resource
def load_detector():
    """Load detector model"""
    return ToxicCommentDetector()

# Main UI
st.markdown("# 🚨 AI Toxic Comment Detector")
st.markdown("### Detect harmful, offensive, and toxic content using Hugging Face Transformers")

# Load detector
detector = load_detector()

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Instructions")
    st.markdown("""
    1. Enter text to analyze
    2. Click "Analyze" button
    3. View detailed results:
       - Toxicity scores
       - Detected categories
       - Token analysis
       - Text features
    """)
    
    st.markdown("---")
    st.markdown("## 🧠 LLM Concepts")
    st.markdown("""
    **Transformer Architecture:**
    - BERT-based model fine-tuned on 159K comments
    
    **Multi-label Classification:**
    - 6 toxicity categories detected simultaneously
    
    **Zero-Shot Learning:**
    - No fine-tuning needed for custom categories
    
    **Tokenization:**
    - Shows how model breaks text into tokens
    
    **Embeddings:**
    - Semantic representation in vector space
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Model Stats")
    st.markdown("""
    **Detoxify Model:**
    - 98.64% AUC on Jigsaw Challenge
    - Trained on 159K Wikipedia comments
    - 6 toxicity categories
    
    **Zero-Shot Classifier:**
    - BART-Large-MNLI
    - No training data needed
    """)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Single Analysis",
    "📊 Batch Analysis",
    "🎓 Learn",
    "📈 Examples",
    "⚙️ Settings"
])

# TAB 1: Single Analysis
with tab1:
    st.subheader("Analyze a Single Comment")
    
    # Input
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste a comment here...",
        height=150,
        max_chars=1000
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    with col3:
        about_button = st.button("ℹ️ About", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if analyze_button and text_input.strip():
        with st.spinner("🔄 Analyzing with LLM..."):
            time.sleep(1)  # Show loading
            analysis = detector.full_analysis(text_input)
        
        # Results header
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Final verdict
        verdict = analysis['final_verdict']
        toxicity_score = verdict['ensemble_toxicity_score']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Toxicity Score",
                f"{toxicity_score:.1%}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Status",
                verdict['recommendation'],
                delta=None
            )
        
        with col3:
            st.metric(
                "Primary Category",
                analysis['zeroshot_results']['primary_category'].title(),
                delta=None
            )
        
        with col4:
            st.metric(
                "Confidence",
                f"{verdict['confidence']:.1%}",
                delta=None
            )
        
        # Show verdict badge
        if verdict['is_toxic']:
            st.markdown('<div class="toxic-badge">⚠️ TOXIC CONTENT DETECTED</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-badge">✅ CONTENT APPEARS SAFE</div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed scores
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🎯 Toxicity Scores")
            detoxify_scores = analysis['detoxify_results']['scores']
            scores_df = pd.DataFrame([detoxify_scores]).T
            scores_df.columns = ['Score']
            scores_df['Score'] = scores_df['Score'].apply(lambda x: f"{x:.1%}")
            
            # Color code
            fig_bar = px.bar(
                x=list(detoxify_scores.keys()),
                y=list(detoxify_scores.values()),
                labels={'x': 'Category', 'y': 'Score'},
                title="Toxicity by Category",
                color=list(detoxify_scores.values()),
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔤 Text Features")
            features = analysis['text_features']
            
            feature_data = {
                'Feature': list(features.keys()),
                'Value': [str(v)[:50] for v in features.values()]
            }
            st.dataframe(
                pd.DataFrame(feature_data),
                use_container_width=True,
                hide_index=True
            )
        
        # Tokenization analysis
        st.markdown("---")
        st.markdown("#### 🔤 Tokenization Analysis (Shows: How Transformer Sees Text)")
        
        tokenization = analysis['tokenization']
        tokens = tokenization['tokens']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tokens", tokenization['num_tokens'])
        with col2:
            st.metric("Max Token Length", tokenization['max_length'])
        with col3:
            st.metric("Avg Tokens/Word", f"{tokenization['num_tokens'] / (tokenization['num_tokens']/len(text_input.split())):.2f}" if text_input.split() else 0)
        
        st.markdown("**Token Breakdown:**")
        tokens_str = " | ".join([f"`{t}`" for t in tokens])
        st.markdown(tokens_str)
        
        # Model info
        st.markdown("---")
        st.markdown("#### 🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Detoxify Model (Primary):**
            - Architecture: BERT-base-uncased
            - Training Data: 159K Wikipedia comments
            - Challenge: Jigsaw 2018
            - AUC Score: 98.64%
            - Categories: 6
            """)
        
        with col2:
            st.markdown("""
            **Zero-Shot Classifier (Secondary):**
            - Architecture: BART-Large-MNLI
            - Training: Trained on NLI data
            - Method: No fine-tuning needed
            - Categories: Custom (6 defined)
            """)
    
    elif analyze_button and not text_input.strip():
        st.error("❌ Please enter text to analyze")
    
    if about_button:
        st.info("""
        This app uses state-of-the-art LLM techniques:
        
        1. **Transformer Models**: BERT architecture fine-tuned on toxic comment dataset
        2. **Multi-label Classification**: Detects 6 types of toxicity simultaneously
        3. **Ensemble Approach**: Combines multiple models for better accuracy
        4. **Zero-Shot Learning**: Classifies without requiring training data
        5. **Tokenization**: Shows how text is processed at token level
        """)

# TAB 2: Batch Analysis
with tab2:
    st.subheader("Analyze Multiple Comments")
    
    batch_text = st.text_area(
        "Enter comments (one per line):",
        placeholder="Comment 1\nComment 2\nComment 3",
        height=200
    )
    
    if st.button("📊 Analyze Batch", use_container_width=True, type="primary"):
        if batch_text.strip():
            texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
            
            with st.spinner(f"Analyzing {len(texts)} comments..."):
                results_df = detector.batch_analyze(texts)
            
            st.markdown("### Results")
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                toxic_count = results_df['is_toxic'].sum()
                st.metric("Toxic Comments", toxic_count)
            
            with col2:
                avg_toxicity = results_df['toxicity_score'].mean()
                st.metric("Avg Toxicity", f"{avg_toxicity:.1%}")
            
            with col3:
                total = len(results_df)
                st.metric("Total Analyzed", total)

# TAB 3: Learn
with tab3:
    st.subheader("🎓 Understanding Toxic Comment Detection")
    
    st.markdown("""
    ### What is Toxic Comment Detection?
    
    Toxic comment detection is a text classification task that identifies harmful, offensive, 
    or abusive language in user-generated content.
    
    ### Why is it important?
    
    - **Online Safety**: Protects communities from harassment and hate speech
    - **Content Moderation**: Helps platforms manage user-generated content
    - **Mental Health**: Prevents cyberbullying and toxic behavior
    - **Fairness**: Ensures equitable online environments
    
    ### How does our model work?
    
    1. **Tokenization**: Text is broken into tokens (words/subwords)
    2. **Embedding**: Each token is converted to a vector
    3. **Transformer Encoding**: BERT processes all tokens with attention
    4. **Classification**: Predicts 6 toxicity categories
    
    ### The Dataset
    
    - **Source**: Jigsaw Toxic Comment Classification Challenge (2018)
    - **Size**: 159,571 comments
    - **Labels**: 6 toxicity types
    - **Data**: Wikipedia Talk Page comments
    
    ### Model Accuracy
    
    - **AUC-ROC**: 98.64% on test set
    - **Precision**: ~96%
    - **Recall**: ~89%
    
    ### Toxicity Categories
    
    1. **Toxic**: Rude, disrespectful, or unreasonable
    2. **Severe Toxic**: Very hateful, aggressive, disrespectful
    3. **Obscene**: Contains obscene language
    4. **Threat**: Contains threats of harm
    5. **Insult**: Direct insults or attacks
    6. **Identity Hate**: Attacks based on identity/attributes
    """)

# TAB 4: Examples
with tab4:
    st.subheader("📈 Example Analyses")
    
    examples = {
        "Non-Toxic (Positive)": "I absolutely love this! Great work everyone!",
        "Non-Toxic (Neutral)": "The weather today is nice.",
        "Slightly Toxic": "This is a stupid idea.",
        "Toxic (Insult)": "You're an idiot, shut up.",
        "Toxic (Severe)": "I hate you, you're worthless!",
    }
    
    selected_example = st.selectbox("Select an example:", list(examples.keys()))
    
    if st.button("🔍 Analyze Selected Example", use_container_width=True, type="primary"):
        example_text = examples[selected_example]
        
        with st.spinner("Analyzing..."):
            time.sleep(1)
            analysis = detector.full_analysis(example_text)
        
        st.markdown(f"**Text:** `{example_text}`")
        
        verdict = analysis['final_verdict']
        st.markdown(f"**Verdict:** {verdict['recommendation']}")
        st.markdown(f"**Toxicity Score:** {verdict['ensemble_toxicity_score']:.1%}")
        st.markdown(f"**Primary Category:** {analysis['zeroshot_results']['primary_category'].title()}")

# TAB 5: Settings
with tab5:
    st.subheader("⚙️ Model Settings")
    
    st.markdown("""
    ### Available Models
    
    #### Detoxify Models:
    - **original**: BERT-base-uncased (98.64% AUC)
    - **unbiased**: RoBERTa-base (93.74% AUC) - Better for identity bias
    - **multilingual**: XLM-RoBERTa (92.11% AUC) - 7 languages
    
    #### Zero-Shot Classifier:
    - facebook/bart-large-mnli (current)
    
    ### Performance Metrics
    
    | Model | Accuracy | Training Data | Languages |
    |-------|----------|---------------|-----------|
    | original | 98.64% | 159K comments | English |
    | unbiased | 93.74% | Bias-focused | English |
    | multilingual | 92.11% | Multi-lang | 7 languages |
    
    ### Citation
    
    ```
    @misc{Detoxify,
      title={Detoxify},
      author={Hanu, Laura and {Unitary team}},
      howpublished={Github. https://github.com/unitaryai/detoxify},
      year={2020}
    }
    ```
    """)

st.markdown("---")
st.markdown("<center>Built with ❤️ using Hugging Face Transformers | 🚨 Toxic Comment Detector</center>", 
           unsafe_allow_html=True)
