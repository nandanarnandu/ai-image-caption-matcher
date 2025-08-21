# 🤖 AI Image Caption Matcher

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-v2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An AI-powered application that generates captions for images and matches them with the most relevant text descriptions. Built with **Python, PyTorch, Hugging Face Transformers, and Flask.**

## ✨ Features

- **🧠 AI-Powered** - Uses OpenAI CLIP for image-text matching
- **⚡ Real-time** - Instant caption matching with confidence scores
- **🎨 Beautiful UI** - Glassmorphism design with animations
- **📱 Responsive** - Works on desktop and mobile

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/nandanarnandu/ai-image-caption-matcher.git
cd ai-image-caption-matcher
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install and run
pip install flask torch torchvision transformers scikit-learn pillow
python app.py

# Open http://localhost:5000
```

## 📸 Usage

1. Upload an image (drag & drop or click)
2. Wait for AI processing (first run downloads model)
3. View top 5 matching captions with confidence scores

## 🛠️ Tech Stack

- **Backend**: Python, Flask, PyTorch
- **AI**: OpenAI CLIP (ViT-B/32)
- **Frontend**: HTML5, CSS3, JavaScript

## 🎨 Customization

Add custom captions in `app.py`:
```python
CANDIDATE_CAPTIONS = [
    "Your custom caption",
    "Another caption",
    # Add more...
]
```

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repo
2. Create feature branch
3. Make changes
4. Submit pull request

---





