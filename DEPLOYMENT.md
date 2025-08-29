# Deployment Checklist

## ✅ Repository Ready
- [x] Git repository initialized
- [x] All files committed
- [x] .gitignore configured
- [x] Streamlit config added

## 📋 GitHub Setup Steps

### 1. Create GitHub Repository
1. Go to [github.com](https://github.com)
2. Click "+" → "New repository"
3. Name: `content-ranker-app` (or your choice)
4. Description: `AI-powered community content ranking with TF-IDF and OpenAI embeddings`
5. Make it **Public** (required for free Streamlit deployment)
6. **Don't initialize** with README (we already have one)
7. Click "Create repository"

### 2. Push to GitHub
```bash
# Replace YOUR_USERNAME and YOUR_REPO_NAME with your actual details
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## 🚀 Streamlit Deployment Steps

### 1. Deploy to Streamlit Community Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Branch: `main`
6. Main file path: `app.py`
7. Click "Deploy"

### 2. Post-Deployment
- Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`
- Update the README.md with the actual URL
- Test both TF-IDF and OpenAI modes

## 🔧 Optional: Environment Variables
If you want to pre-configure an OpenAI API key for the deployment:
1. In Streamlit Cloud, go to your app settings
2. Add secrets: `OPENAI_API_KEY = "your_key_here"`
3. The app will automatically use it

## 📝 Update URLs
After deployment, update these files with your actual URLs:
- README.md: Replace placeholder URLs with real deployment URL
- Add GitHub repository URL to README.md

## 🎯 Success Criteria
- [x] App builds successfully
- [ ] App deploys without errors
- [ ] TF-IDF mode works
- [ ] OpenAI mode works with API key input
- [ ] All ranking features functional
- [ ] Mobile responsive design works
