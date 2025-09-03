# Content Ranking Algorithm Documentation

## 🎯 Overview

The Community Content Ranker uses a sophisticated **multi-factor scoring system** that combines three different dimensions to evaluate and rank content quality. This approach mimics how humans naturally assess content relevance, usefulness, and credibility.

## 📊 The Three Ranking Factors

### 1. 🔍 Relevance Score (0.0 - 1.0)

**Purpose**: Measures how semantically similar the content is to the user's query

**Implementation**:
- **TF-IDF Mode**: Uses Term Frequency-Inverse Document Frequency vectorization
- **OpenAI Mode**: Uses advanced text embeddings for semantic understanding
- **Process**:
  1. Converts all posts and user query into numerical vectors
  2. Calculates cosine similarity between query vector and each post vector
  3. Normalizes similarity scores to 0-1 range

**Technical Details**:
```python
# TF-IDF Implementation
X, vec = embed_texts_tfidf(texts)
qX, _ = embed_texts_tfidf([query], fit_vectorizer=vec)
rel = cosine_similarity(X, qX).squeeze()

# OpenAI Implementation  
embs = embed_texts_openai(texts + [query], openai_api_key)
post_embs = embs[:-1]
query_emb = embs[-1:]
rel = (post_embs @ query_emb.T).squeeze()
```

**Example**:
- Query: "beginner investing advice"
- High relevance: "How to start investing with $100 per month"
- Low relevance: "Student loan repayment strategies"

### 2. 📚 Helpfulness Score (0.0 - 1.0)

**Purpose**: Evaluates how actionable and practical the content is

**Scoring Components**:

#### Numbers/Data (50% weight)
- Counts all digits in the text
- More specific numbers indicate more concrete, actionable advice
- Formula: `min(num_tokens / 10.0, 1.0) * 0.5`

#### Step-by-step Structure (30% weight)
- Detects structured guidance patterns
- Keywords: "step", "1)", "2)", "first,", "second,", "third,", "bullet", "- "
- Binary scoring: 1.0 if detected, 0.0 otherwise

#### Actionable Language (20% weight)
- Identifies instructional and helpful phrases
- Keywords: "how to", "you can", "do this", "here's how", "here is how", "here is what to do"
- Binary scoring: 1.0 if detected, 0.0 otherwise

**Formula**:
```python
score = 0.0
score += min(num_tokens / 10.0, 1.0) * 0.5      # Numbers
score += (1.0 if has_steps else 0.0) * 0.3      # Structure
score += (1.0 if has_howto else 0.0) * 0.2      # Actionable language
```

**Examples**:
- **High score**: "Here's how to start: 1) Open a Roth IRA, 2) Choose index funds with 0.05% expense ratio, 3) Set up automatic $100 monthly contributions"
- **Low score**: "Investing is good for your future"

### 3. 🛡️ Trustworthiness Score (0.0 - 1.0)

**Purpose**: Assesses content credibility and reliability

**Scoring Components**:

#### Credible Sources (50% weight)
- Detects authoritative domain references
- Keywords: ".gov", ".edu"
- Binary scoring: 0.5 if detected, 0.0 otherwise

#### Financial Expertise (50% weight)
- Counts domain-specific financial terminology
- Keywords: "apr", "apy", "roi", "brokerage", "index fund", "expense ratio", "etf", "401k", "ira", "roth", "capital gains", "dividend", "rebalance", "budget", "cash flow", "savings rate", "debt", "credit score", "emergency fund", "compound", "interest", "tax", "treasury", "cd", "certificate of deposit", "dca", "dollar-cost averaging"
- Formula: `min(has_keywords / 5.0, 1.0) * 0.5`

#### Hype Penalty
- Identifies and penalizes suspicious language
- Keywords: "guaranteed", "moon", "to the moon", "risk-free", "secret", "trick", "hack", "get rich quick", "insider"
- Penalty: `min(hype_hits * 0.25, 0.75)`

**Formula**:
```python
score = 0.0
score += 0.5 if has_gov_edu else 0.0                    # Credible sources
score += min(has_keywords / 5.0, 1.0) * 0.5            # Financial expertise
score -= min(hype_hits * 0.25, 0.75)                   # Hype penalty
return max(0.0, min(1.0, score))                       # Clamp to [0,1]
```

**Examples**:
- **High score**: "According to IRS.gov, Roth IRA contributions have income limits. The expense ratio is a key factor in ETF selection."
- **Low score**: "This guaranteed risk-free investment secret will make you rich quick! Insider trick to the moon!"

## ⚖️ Final Scoring Formula

The final ranking score combines all three factors using user-adjustable weights:

```
Final Score = (Relevance × W₁) + (Helpfulness × W₂) + (Trustworthiness × W₃)
```

Where:
- **W₁** = Relevance weight (default: 0.5)
- **W₂** = Helpfulness weight (default: 0.25)  
- **W₃** = Trustworthiness weight (default: 0.25)
- All individual scores are normalized to [0,1] range
- Final scores determine the ranking order (descending)

## 🎛️ Weight Configuration Examples

### Relevance Focus (0.8, 0.1, 0.1)
- **Use case**: Finding content that directly matches the query
- **Result**: Prioritizes semantic similarity over content quality
- **Best for**: Specific topic searches

### Balanced (0.5, 0.25, 0.25)
- **Use case**: General content discovery
- **Result**: Equal consideration of all factors
- **Best for**: Most use cases

### Trust & Help Focus (0.2, 0.4, 0.4)
- **Use case**: Quality content curation
- **Result**: Prioritizes helpful, credible content over relevance
- **Best for**: Educational content, expert recommendations

## 🔧 Technical Implementation

### Embedding Methods

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- **Pros**: Fast, no API costs, works offline
- **Cons**: Keyword-based, limited semantic understanding
- **Performance**: ~100ms per query
- **Use case**: Default mode, high-volume applications

#### OpenAI Embeddings
- **Pros**: Superior semantic understanding, context awareness
- **Cons**: API costs, requires internet connection
- **Performance**: ~1-2s per query (API dependent)
- **Cost**: ~$0.10 per million tokens
- **Use case**: High-quality applications, semantic search

### Normalization Process

All scores are normalized to ensure fair comparison:

```python
# Relevance normalization
if np.max(rel) > np.min(rel):
    rel_norm = (rel - np.min(rel)) / (np.max(rel) - np.min(rel))
else:
    rel_norm = np.zeros_like(rel)
```

### Error Handling

The system includes robust fallback mechanisms:
- OpenAI API failures automatically fall back to TF-IDF
- Invalid API keys gracefully degrade to TF-IDF mode
- Missing data is handled with default values

## 🧠 Algorithm Rationale

This multi-factor approach is designed to replicate human content evaluation patterns:

1. **Relevance**: "Does this answer my question?"
2. **Helpfulness**: "Is this information useful and actionable?"
3. **Trustworthiness**: "Can I trust this source and information?"

### Why This Works

- **Prevents gaming**: Multiple factors make it harder to manipulate rankings
- **Quality focus**: Helps surface high-value content over clickbait
- **User control**: Adjustable weights allow customization for different use cases
- **Scalable**: Efficient algorithms handle large content volumes
- **Explainable**: Clear scoring breakdown helps users understand rankings

## 📈 Performance Characteristics

### Computational Complexity
- **TF-IDF**: O(n×m) where n = posts, m = vocabulary size
- **OpenAI**: O(n) API calls + O(n×d) similarity computation
- **Heuristics**: O(n×k) where k = average text length

### Scalability
- **TF-IDF**: Handles thousands of posts efficiently
- **OpenAI**: Limited by API rate limits (~60 requests/minute)
- **Memory**: Minimal memory footprint for typical datasets

### Accuracy Trade-offs
- **TF-IDF**: Good for keyword matching, poor for semantic similarity
- **OpenAI**: Excellent semantic understanding, requires API costs
- **Heuristics**: Fast and reliable, but rule-based limitations

## 🔮 Future Enhancements

### Machine Learning Integration
- Replace rule-based heuristics with trained models
- Use user feedback to improve ranking accuracy
- Implement learning-to-rank algorithms

### Advanced Features
- **Temporal scoring**: Boost recent content
- **User personalization**: Adapt to individual preferences
- **A/B testing**: Compare ranking strategies
- **Real-time updates**: Dynamic content ingestion

### Performance Optimizations
- **Caching**: Store embeddings for repeated queries
- **Batch processing**: Handle multiple queries efficiently
- **CDN integration**: Global content delivery
- **Database optimization**: Efficient storage and retrieval

---

*This algorithm represents a production-ready approach to content ranking that balances accuracy, performance, and user control while maintaining explainability and extensibility.*
