# SCRUM-CLI

AI meeting assistant that transcribes audio and roasts your corporate buzzwords.

## What it does

- Transcribes meeting audio with speaker names
- Interactive AI chat about your meetings  
- `/ridiculous` mode roasts meeting culture with buzzword counts
- Vector search across all meetings
- Real-time transcription
<br>

<img src = image.png>
<br>

## Install & Run

# Support for Python3.9 only! 🙇 Apologies for setup

### Quick Install with uv (Recommended)
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/Rachit-Gandhi/scrum-cli.git
cd scrum-cli
uv sync
cp .env.example .env
# Add your API keys to .env
uv run scrum-cli transcribe meeting.mp3 --chat
```

### Traditional pip install - bound to fail - barely works
```bash
pip install scrum-cli
cp .env.example .env
# Add your API keys to .env
scrum-cli transcribe meeting.mp3 --chat
```

## API Keys Needed

Get these and put in `.env`:
- `GEMINI_API_KEY` from [Google AI Studio](https://ai.google.dev/)
- `HUGGING_FACE_TOKEN` from [HuggingFace](https://huggingface.co/settings/tokens)

**⚠️ Except HACKATHON JUDGES: Please use your own API keys. The ngrok tunnel below will be updated during judging.**

## Live Demo Access

**Ngrok URL:** [WILL BE UPDATED FOR JUDGING]
```
https://695261c929e5.ngrok-free.app
```

## Chat Commands

```
/ridiculous   # Roast mode with buzzword analysis
/summary      # Meeting summary
/actionitems  # Extract action items  
/stats        # Speaker stats
/whosaid      # Search who said what
```

## Example Output

```
🎭 ROAST ANALYSIS
├── 🎯 Buzzwords: "synergy" (7x), "leverage" (12x)  
├── 🗣️ Mike dominated 67% of airtime
├── 📝 Grade: D+ ("More buzzwords than decisions")
└── 🏆 "Let's leverage synergistic solutions"
```

## ⚙️ Requirements

- **Python 3.9+**
- **Required API Keys:**
  - [Google Gemini API key](https://ai.google.dev/) (required)
  - [HuggingFace token](https://huggingface.co/settings/tokens) (for speaker diarization)

## 📖 Usage Examples

### Basic Transcription
```bash
# Transcribe an audio file
scrum-cli transcribe meeting.mp3

# With speaker diarization
scrum-cli transcribe meeting.mp3 --speakers
```

### Interactive Chat Interface
```bash
# Start transcription with chat interface
scrum-cli transcribe meeting.mp3 --chat
# Using the ngrok tunnel
scrum-cli stream Product_Meeting.mp3 -p https://695261c929e5.ngrok-free.app -s 2
```

### Available Chat Commands

```bash
/help         # Show all commands
/summary      # Professional meeting summary  
/ridiculous   # Hilarious roast analysis 🎭
/actionitems  # Extract action items
/decisions    # Show decisions made
/stats        # Speaker statistics
/whosaid      # Find who said what
/live         # Recent transcription (during streaming)
/export       # Save chat history
/quit         # Exit
```

### Chat Examples

```
> Who talked the most in the meeting?
🤖 Mike dominated with 67% of speaking time (12.4 minutes)

> /ridiculous
🎭 ROAST MODE ACTIVATED! 
*Generates hilarious analysis with buzzword counts and witty observations*

> What were the main action items?
📋 Found 3 action items:
1. Sarah will update the budget by Friday
2. Mike to schedule follow-up with clients  
3. Team to review proposal draft
```

## 🏗️ Architecture

```
scrum-cli/
├── scrum_cli/              # Main package
│   ├── main.py            # CLI entry point
│   ├── chatbot_ui.py      # Rich TUI chat interface  
│   ├── live_transcription.py  # Real-time audio processing
│   ├── memory_store.py    # Vector database integration
│   ├── proxy_server.py    # FastAPI backend for AI
│   ├── roast_engine.py    # Meeting culture analysis
│   └── ngrok_manager.py   # Tunnel management
└── transcription-tool/     # Audio processing utilities
```

## 🎯 Features in Detail

### Real-Time Transcription
- Supports multiple audio formats (MP3, WAV, M4A, etc.)
- Advanced speaker diarization using PyAnnote
- Streaming transcription with live updates
- Meeting data persistence with vector search

### AI-Powered Analysis  
- **Professional Mode**: Extract action items, decisions, and summaries
- **Roast Mode**: Hilarious analysis of meeting culture and patterns
- **Natural Language Queries**: Ask questions about meeting content
- **Contextual Memory**: Remembers past meetings and speaker preferences

### Meeting Analytics
- Speaker time distribution and statistics
- Buzzword detection and counting  
- Meeting efficiency scoring
- Personality profile generation
- Export capabilities for further analysis


### Project Structure

The project follows a modular architecture:
- `scrum_cli/` - Main package with core functionality
- `transcription-tool/` - Audio processing and transcription utilities  
- Tests and development files are hidden via `.gitignore`

## 📦 PyPI Deployment

**Note**: PyPI upload is currently in progress. For now, please use the local installation method with `uv` or `pip` from the source code.

The project is configured for PyPI publication with:
- Proper `setup.py` and `pyproject.toml` configuration
- Version management and dependency specifications
- Comprehensive metadata and classifiers
- Development dependencies separation
- GitHub Actions workflow for automated publishing

Sorry for the inconvenience - we're working on getting this published to PyPI soon!

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Community

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Rachit-Gandhi/scrum-cli/issues)
- 💬 **Feature Requests**: [GitHub Discussions](https://github.com/Rachit-Gandhi/scrum-cli/discussions)  
- 📧 **Contact**: Create an issue for support

## 🎉 Acknowledgments

- Built with [Rich](https://github.com/Textualize/rich) for beautiful terminal UI
- Powered by [Google Gemini](https://ai.google.dev/) for AI capabilities
- Uses [PyAnnote](https://github.com/pyannote/pyannote-audio) for speaker diarization
- Audio processing with [Whisper](https://github.com/openai/whisper) and [Pydub](https://github.com/jiaaro/pydub)

---

*Built with ❤️ for developers who believe meetings should be both productive and entertaining.*

**Try `/ridiculous` mode - you won't regret it!** 🎭