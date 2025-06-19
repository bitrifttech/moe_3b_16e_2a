"""
Wikipedia Dataset Loader

Loads and processes Wikipedia articles for knowledge training.
"""

from typing import List, Dict, Any
from datasets import load_dataset as hf_load_dataset
from ..base import BaseDatasetLoader, DatasetConfig

class WikipediaLoader(BaseDatasetLoader):
    """Loader for Wikipedia dataset."""
    
    def __init__(self, config: DatasetConfig = None, language: str = "en", date: str = "20220301"):
        if config is None:
            config = DatasetConfig(
                name="Wikipedia",
                max_samples=50000,
                max_length=600,
                min_length=100
            )
        super().__init__(config)
        self.language = language
        self.date = date
    
    def load_raw_data(self) -> Any:
        """Load Wikipedia dataset from HuggingFace."""
        try:
            # Load a 1% slice of the dataset for the specified language and date
            dataset = hf_load_dataset("wikipedia", f"{self.date}.{self.language}", split=f"train[:{self.percent_to_load}%]")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load Wikipedia dataset: {e}")
            raise
    
    def preprocess(self, raw_data: Any) -> List[Dict[str, str]]:
        """
        Process Wikipedia data into article format.
        
        Extracts article text and cleans it up.
        """
        processed_data = []
        
        for item in raw_data:
            title = item.get("title", "").strip()
            text = item.get("text", "").strip()
            
            if not text or not title:
                continue
            
            # Clean up the text
            cleaned_text = self._clean_wikipedia_text(text)
            
            # Format with title
            article_text = f"# {title}\n\n{cleaned_text}"
            
            processed_data.append({
                "text": article_text,
                "metadata": {
                    "source": "wikipedia",
                    "title": title,
                    "language": self.language,
                    "type": "encyclopedia_article"
                }
            })
        
        return processed_data
    
    def _clean_wikipedia_text(self, text: str) -> str:
        """Clean Wikipedia text of formatting artifacts."""
        import re
        
        # Remove common Wikipedia artifacts
        text = re.sub(r'\{\{[^}]*\}\}', '', text)  # Remove templates
        text = re.sub(r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2', text)  # Clean links
        text = re.sub(r'\[http[^\]]*\]', '', text)  # Remove external links
        text = re.sub(r'==+\s*([^=]+)\s*==+', r'\n\n## \1\n\n', text)  # Format headers
        text = re.sub(r'\n\n+', '\n\n', text)  # Normalize whitespace
        text = re.sub(r'^\s*\*', '•', text, flags=re.MULTILINE)  # Clean bullet points
        
        return text.strip()
    
    def _apply_custom_filters(self, item: Dict[str, str]) -> bool:
        """Apply Wikipedia-specific filters."""
        text = item["text"]
        title = item["metadata"].get("title", "")
        
        # Filter out disambiguation pages
        if "disambiguation" in title.lower() or "may refer to" in text.lower():
            return False
        
        # Filter out list articles (usually not great for training)
        if title.lower().startswith("list of"):
            return False
        
        # Filter out very short articles
        if len(text.split()) < 50:
            return False
        
        # Filter out articles with too many special characters (likely formatting issues)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-()[]{}":;\n#•') / len(text)
        if special_char_ratio > 0.15:
            return False
        
        return True

def create_wikipedia_loader(
    language: str = "en", 
    date: str = "20220301",
    max_samples: int = 50000, 
    max_length: int = 600,
    percent_to_load: int = 1
) -> WikipediaLoader:
    """Factory function to create Wikipedia loader with custom config."""
    config = DatasetConfig(
        name=f"Wikipedia-{language}",
        max_samples=max_samples,
        max_length=max_length,
        min_length=100
    )
    loader = WikipediaLoader(config, language, date)
    loader.percent_to_load = percent_to_load
    return loader
