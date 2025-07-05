**Steps:**

1. **Create a GitHub repository** and push your code
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Sign in with GitHub**
4. **Click "New app"**
5. **Select your repository**
6. **Set main file path:** `streamlit_app.py`
7. **Click "Deploy"**

**App will be live at:** `https://your-app-name.streamlit.app`

### 2. 🎨 Hugging Face Spaces

**Steps:**

1. **Go to [huggingface.co/spaces](https://huggingface.co/spaces)**
2. **Click "Create new Space"**
3. **Choose "Streamlit" as SDK**
4. **Upload your files**
5. **App will be live automatically**

## 📁 Required Files for Deployment

Make sure these files are in your repository:

```
NIRF/
├── streamlit_app.py          # Main app file
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── config.toml          # Streamlit config
├── data/
│   └── raw/                 # Data files (optional)
└── README.md
```

## 🔧 Quick Deployment Steps

### Option A: Streamlit Cloud (Easiest)

1. **Create GitHub Repository:**

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/nirf-dashboard.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file: `streamlit_app.py`
   - Deploy!

### Option B: Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```
