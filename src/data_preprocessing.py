"""
Scientific Text Preprocessing Module

This module handles specialized text preprocessing for scientific literature,
with specific handling for:
- Chemical formulas
- Mathematical notations
- Technical abbreviations
- Domain-specific stopwords

Classes:
    ScientificTextPreprocessor: Main class for text preprocessing operations
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class ScientificTextPreprocessor:
    """
    A text preprocessor specialized for scientific literature.
    
    Handles specialized cleaning of scientific texts including:
    - Chemical formula preservation
    - Technical term handling
    - Scientific stopwords management
    - Sub/superscript normalization
    
    Attributes:
        stop_words (set): Customized stopwords excluding scientific terms
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer instance
        formula_pattern (re.Pattern): Regex for chemical formula detection
    """

    def __init__(self):
        """Initialize the preprocessor with custom scientific settings."""
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        # Keep important scientific terms
        self.stop_words -= {'using', 'due', 'based', 'including', 'used', 'show', 'may', 'could'}
        self.lemmatizer = WordNetLemmatizer()
        self.formula_pattern = re.compile(r'<SUB>.*?</SUB>|<SUP>.*?</SUP>')
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned and processed text
            
        Example:
            >>> preprocessor = ScientificTextPreprocessor()
            >>> preprocessor.clean_text("The H2O molecule's structure...")
            'h2o molecule structure'
        """
        # Handle missing/NaN values
        if not isinstance(text, str):
            return ""
            
        # Step 1: Handle chemical formulas - replace with plain text
        text = self.formula_pattern.sub(' ', text)
        
        # Step 2: Basic cleaning
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Replace non-letters with space
        text = re.sub(r'\s+', ' ', text).strip()   # Normalize whitespace
        
        # Step 3: Token processing
        tokens = text.split()
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words:
                # Special handling for common scientific suffixes
                if token.endswith(('ing', 'tion', 'ment')):
                    token = self.lemmatizer.lemmatize(token, pos='v')
                else:
                    token = self.lemmatizer.lemmatize(token)
                processed_tokens.append(token)
                
        return ' '.join(processed_tokens)
    
    def preprocess_data(self, df):
        """
        Preprocess an entire DataFrame containing scientific texts.
        
        Args:
            df (pd.DataFrame): DataFrame with 'title' and/or 'abstract' columns
            
        Returns:
            pd.DataFrame: Processed DataFrame with new 'cleaned_text' column
            
        Raises:
            ValueError: If neither 'title' nor 'abstract' columns are present
        """
        # Create a copy to avoid modifying the original DataFrame
        processed_df = df.copy()
        
        # Combine title and abstract if they exist (handle NaN values)
        if 'title' in df.columns and 'abstract' in df.columns:
            processed_df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
        elif 'title' in df.columns:
            processed_df['text'] = df['title'].fillna('')
        elif 'abstract' in df.columns:
            processed_df['text'] = df['abstract'].fillna('')
        else:
            raise ValueError("DataFrame must contain either 'title' or 'abstract' columns")
        
        # Clean the text (handle NaN values)
        processed_df['cleaned_text'] = processed_df['text'].apply(
            lambda x: self.clean_text(x) if isinstance(x, str) else ""
        )
        
        return processed_df