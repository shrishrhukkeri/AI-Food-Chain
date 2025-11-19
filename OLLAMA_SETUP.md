# 🤖 Ollama Setup Guide

The chatbot now uses **Ollama (local AI) as the first preference**, with Gemini API as a fallback. This means you can run the AI chatbot completely offline and locally!

## Quick Setup

### Step 1: Install Ollama

**macOS:**
```bash
brew install ollama
# Or download from: https://ollama.ai/download
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from: https://ollama.ai/download

### Step 2: Start Ollama

```bash
ollama serve
```

This will start Ollama on `http://localhost:11434` (default port).

### Step 3: Pull a Model

Pull a model that supports your languages (English, Hindi, Kannada):

```bash
# Recommended: Llama 3.2 (good multilingual support)
ollama pull llama3.2

# Alternative options:
# ollama pull llama3.1
# ollama pull mistral
# ollama pull qwen2.5
```

### Step 4: Configure (Optional)

The `.env` file already has default settings:
```
OLLAMA_MODEL=llama3.2
OLLAMA_URL=http://localhost:11434
```

You can change these if needed:
- **OLLAMA_MODEL**: The model name you pulled (e.g., `llama3.2`, `llama3.1`, `mistral`)
- **OLLAMA_URL**: The URL where Ollama is running (default: `http://localhost:11434`)

### Step 5: Restart Flask App

After setting up Ollama, restart your Flask application:

```bash
python app.py
```

## How It Works

1. **First Attempt**: The chatbot tries to use Ollama (local)
   - If Ollama is running and the model is available, it uses it
   - Fast, private, and works offline!

2. **Fallback**: If Ollama is not available, it falls back to Gemini API
   - Requires `GEMINI_API_KEY` in `.env`
   - Works over the internet

3. **Final Fallback**: If both fail, it uses rule-based responses
   - Answers basic questions using farm data
   - No AI, but still helpful

## Testing

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Test if the model is available:
   ```bash
   ollama run llama3.2 "Hello, how are you?"
   ```

3. Open the dashboard and try the chatbot
   - It should use Ollama automatically!

## Troubleshooting

### "Ollama connection failed"
- Make sure Ollama is running: `ollama serve`
- Check if it's on the correct port: `http://localhost:11434`
- Verify the model is pulled: `ollama list`

### "Model not found"
- Pull the model: `ollama pull llama3.2`
- Check available models: `ollama list`
- Update `OLLAMA_MODEL` in `.env` to match your model name

### Slow responses
- Try a smaller/faster model: `ollama pull llama3.2:1b` (1B parameter version)
- Or use a more powerful model if you have the resources

### Model doesn't support Hindi/Kannada well
- Try a different model with better multilingual support:
  - `ollama pull qwen2.5` (good for Asian languages)
  - `ollama pull mistral` (good multilingual support)

## Model Recommendations

| Model | Size | Speed | Multilingual | Best For |
|-------|------|-------|--------------|----------|
| llama3.2 | ~2GB | Fast | Good | General use, English |
| llama3.2:1b | ~700MB | Very Fast | Good | Quick responses |
| qwen2.5 | ~2GB | Fast | Excellent | Hindi, Kannada, Asian languages |
| mistral | ~4GB | Medium | Excellent | Best multilingual support |

## Benefits of Using Ollama

✅ **Privacy**: All processing happens locally  
✅ **Offline**: Works without internet  
✅ **Free**: No API costs  
✅ **Fast**: No network latency  
✅ **Customizable**: Use any model you want  

---

**Need help?** Check the main README.md or open an issue on the project repository.

