import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Create sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()


def preprocess_text(text):
    """
    Perform basic text preprocessing.
    """

    # Remove URLs
    text_without_urls = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Process text using spaCy
    doc = nlp(text_without_urls)

    # Tokenization
    tokens = []

    for token in doc:
        if not token.is_space:
            tokens.append(token.text)

    # Stop-word removal
    without_stopwords = []

    for token in doc:
        if not token.is_space and not token.is_stop:
            without_stopwords.append(token.text)

    # Lemmatization
    lemmas = []

    for token in doc:
        if not token.is_space and not token.is_stop:
            lemmas.append(token.lemma_)

    # Cleaned text
    cleaned_text = " ".join(lemmas)

    return {
        "tokens": tokens,
        "without_stopwords": without_stopwords,
        "lemmas": lemmas,
        "cleaned_text": cleaned_text
    }


def extract_entities(text):
    """
    Extract named entities from the text.
    """

    doc = nlp(text)

    entities = []

    for entity in doc.ents:

        entities.append({
            "text": entity.text,
            "label": entity.label_,
            "description": spacy.explain(entity.label_)
        })

    return entities


def analyze_sentiment(text):
    """
    Analyze sentiment using VADER.
    """

    scores = sentiment_analyzer.polarity_scores(text)

    compound_score = scores["compound"]

    if compound_score >= 0.05:
        label = "POSITIVE"

    elif compound_score <= -0.05:
        label = "NEGATIVE"

    else:
        label = "NEUTRAL"

    return {
        "label": label,
        "compound": scores["compound"],
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"]
    }


def process_text(text):
    """
    Complete NLP pipeline.
    """

    preprocessing = preprocess_text(text)

    entities = extract_entities(text)

    sentiment = analyze_sentiment(text)

    return {
        "original_text": text,
        "cleaned_text": preprocessing["cleaned_text"],
        "tokens": preprocessing["tokens"],
        "without_stopwords": preprocessing["without_stopwords"],
        "lemmas": preprocessing["lemmas"],
        "entities": entities,
        "sentiment": sentiment
    }