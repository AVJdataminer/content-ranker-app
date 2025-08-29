# Ask Dolly — Community Content Ranker

A Streamlit web application that intelligently ranks community posts by **relevance**, **helpfulness**, and **trustworthiness**. This demo showcases advanced content ranking algorithms with both TF-IDF and OpenAI embeddings support.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🚀 Live Demo

**[View Live App](https://your-app-url.streamlit.app)** *(Will be available after deployment)*

## ✨ Features

- **🔍 Dual Embedding Support**: TF-IDF (free, fast) or OpenAI embeddings (accurate, semantic)
- **📊 Multi-Factor Ranking**: Combines relevance, helpfulness, and trustworthiness scores
- **⚖️ Adjustable Weights**: Interactive sliders to customize ranking priorities
- **🔑 Easy API Integration**: Simple UI for adding OpenAI API keys
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🎯 How It Works

### Ranking Factors
1. **Relevance**: Semantic similarity to user query using TF-IDF or OpenAI embeddings
2. **Helpfulness**: Detects actionable advice, step-by-step guidance, and concrete information
3. **Trustworthiness**: Favors credible sources (.gov/.edu links, financial terms) and penalizes hype words

### Scoring Algorithm
Final Score = (Relevance × W₁) + (Helpfulness × W₂) + (Trustworthiness × W₃)

Where W₁, W₂, W₃ are user-adjustable weights.

## 🛠️ Local Development

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/content_ranker_starter.git
cd content_ranker_starter

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Optional: OpenAI Integration

For enhanced semantic understanding, add your OpenAI API key:

1. Get an API key from [OpenAI Platform](https://platform.openai.com)
2. Either:
   - Set environment variable: `export OPENAI_API_KEY=your_key_here`
   - Or enter it directly in the app's sidebar

## 📁 Project Structure

```
content_ranker_starter/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/
│   └── sample_posts.json # Sample community posts
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web app framework
- **scikit-learn**: TF-IDF vectorization and similarity
- **OpenAI**: Advanced embeddings (optional)
- **pandas/numpy**: Data manipulation
- **pathlib**: File handling

### Performance
- **TF-IDF Mode**: ~100ms per query (10 posts)
- **OpenAI Mode**: ~1-2s per query (API dependent)
- **Cost**: OpenAI embeddings ~$0.10 per million tokens

## 🚀 Deployment

### Streamlit Community Cloud
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Docker (Alternative)
```bash
# Build image
docker build -t content-ranker .

# Run container
docker run -p 8501:8501 content-ranker
```

## 🧪 Example Queries

Try these queries to see the ranking in action:

- "Beginner investing advice for recent graduates"
- "Student loan repayment strategies"
- "Building credit score safely"
- "Emergency fund planning"

## 📈 Future Enhancements

- **Machine Learning**: Replace heuristics with trained models
- **User Feedback**: Collect rankings to improve algorithms
- **Real-time Data**: Connect to live community APIs
- **A/B Testing**: Compare ranking strategies
- **Analytics Dashboard**: Track engagement metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Aiden Johnson**
- Email: [aiden.dataminer@gmail.com](mailto:aiden.dataminer@gmail.com)
- GitHub: [@yourusername](https://github.com/yourusername)

---

*Created for demonstration purposes - August 29, 2025*
