# Audio File Transcript

Transcribe audio and video files using Google's Gemini API with automatic speaker diarization, parallel processing, and optional English translation. Works on macOS, Windows, and Linux.

## Features

- **Interactive File Selection** - Choose which files to transcribe
- **Universal Media Support** - Works with audio AND video files
- **Parallel Processing** - Process 5 chunks simultaneously for 2-5x speedup
- **Speaker Diarization** - Automatically identifies and labels different speakers
- **English Translation** - Optionally translate the transcript to English (separate file, 1:1 line mapping)
- **Smart Conversion** - Auto-converts video/audio to optimal format (saves ~90% file size)
- **Secure** - API key saved to .env file, never hardcoded
- **Auto Cleanup** - Removes temporary files automatically
- **Cross-Platform** - Works on macOS, Windows, and Linux

## Quick Start

### macOS / Linux

```bash
# One-time setup
chmod +x setup.sh && ./setup.sh

# Run
python3 transcribe_audio.py
```

### Windows

```powershell
# One-time setup (double-click or run in terminal)
.\setup.bat

# Run
py transcribe_audio.py
```

The script will:
1. Show all media files in the current directory
2. Let you choose which ones to transcribe
3. Ask for output language (Original or Original + English translation)
4. Convert files to optimal format if needed
5. Process them with parallel workers
6. Save transcripts with speaker diarization (original saved first, then translation)
7. Clean up temporary files

## Supported Formats

**Audio:** MP3, M4A, WAV, AAC, OGG, FLAC, WMA
**Video:** MP4, MOV, AVI, MKV, WEBM, FLV (extracts audio automatically)

## Setup

### Option A: Automated Setup (Recommended)

**macOS / Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```powershell
.\setup.bat
```

The setup script will:
- Check that Python 3 is installed
- Fix PATH issues automatically (Windows)
- Install pip if missing
- Install all Python dependencies
- Check for ffmpeg and offer to install it (Windows: via winget)
- Check for API key configuration

### Option B: Manual Setup

**1. Install Python 3**
- macOS: `brew install python3` or download from https://www.python.org/downloads/
- Windows: Download from https://www.python.org/downloads/ (check "Add Python to PATH")
- Linux: `sudo apt install python3 python3-pip`

**2. Install Python dependencies**
```bash
# macOS / Linux
pip3 install -r requirements.txt

# Windows
py -m pip install -r requirements.txt
```

**3. Install ffmpeg**
```bash
# macOS
brew install ffmpeg

# Windows (pick one)
winget install ffmpeg
choco install ffmpeg

# Linux
sudo apt install ffmpeg        # Debian/Ubuntu
sudo dnf install ffmpeg        # Fedora
sudo pacman -S ffmpeg          # Arch
```

**4. Get your API key**

Get it from: https://aistudio.google.com/app/apikey

The script will prompt you on first run and offer to save it to `.env` automatically.

Or create `.env` manually:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Interactive Mode (Recommended)
```bash
# macOS / Linux
python3 transcribe_audio.py

# Windows
py transcribe_audio.py
```

```
================================================================================
MEDIA FILES FOUND
================================================================================
  [1] interview.mp3      MP3   45.2MB
  [2] meeting.m4a        M4A   38.7MB [will convert to MP3]
  [3] video.mp4          MP4  156.3MB [will convert to MP3]
================================================================================

Your selection: 1,2       (or 'all' for all files, 'q' to quit)

================================================================================
OUTPUT LANGUAGE
================================================================================
  [1] Original (keep original language)
  [2] Original with English translation
================================================================================

Your selection: 2
```

### Process a Specific File
```bash
# macOS / Linux
python3 transcribe_audio.py "my_recording.mp3"

# Windows
py transcribe_audio.py "my_recording.mp3"
```

## Output

Each file produces a transcript saved as `filename_transcript_parallel.txt`.

When English translation is selected, a second file `filename_transcript_parallel_en.txt` is created alongside the original. The original is saved first, then the translation — so if translation fails, you still have the original.

**Original transcript** (`interview_transcript_parallel.txt`):
```
================================================================================
MEETING TRANSCRIPT (Gemini Parallel Processing)
================================================================================
Source File: interview.mp3
Duration: 2700.0s (45.0min)
Speakers: 2: Speaker 1, Speaker 2
Processing: 10-min chunks, 5 parallel workers
Total Processing Time: 180.5s (3.0min)
Speedup vs Sequential: 3.2x
================================================================================

[00:00] Speaker 1: Buna ziua, bine ati venit la interviu...

[00:05] Speaker 2: Multumesc ca m-ati invitat...
```

**English translation** (`interview_transcript_parallel_en.txt`):
```
================================================================================
MEETING TRANSCRIPT (Gemini Parallel Processing)
================================================================================
Source File: interview.mp3
Duration: 2700.0s (45.0min)
Speakers: 2: Speaker 1, Speaker 2
Processing: 10-min chunks, 5 parallel workers
Total Processing Time: 180.5s (3.0min)
Speedup vs Sequential: 3.2x
================================================================================

[00:00] Speaker 1: Good day, welcome to the interview...

[00:05] Speaker 2: Thank you for having me...
```

## Configuration

Edit these values in `transcribe_audio.py`:

```python
CHUNK_DURATION_MINUTES = 10  # Duration of each chunk
OVERLAP_SECONDS = 3          # Overlap between chunks
MAX_CONCURRENT_CHUNKS = 5    # Parallel workers
MP3_BITRATE = "64k"          # Conversion quality
MP3_SAMPLE_RATE = "16000"    # 16kHz (matches Gemini)
```

## Why These Settings?

**5 Parallel Chunks** — Conservative limit that works reliably on paid tier accounts while providing 2-5x speedup.

**64kbps / 16kHz / Mono** — Gemini downsamples to 16kbps internally, so higher quality just wastes bandwidth and costs more. These settings reduce file size by ~90% with no impact on transcription quality.

**10-Minute Chunks** — Prevents quality degradation (repetition loops, timestamp drift) that occurs after 18 minutes of continuous audio.

## Troubleshooting

**"pydub not installed" or "audioop" error (Python 3.13+)**
```bash
# macOS / Linux
pip3 install -r requirements.txt

# Windows
py -m pip install -r requirements.txt
```
The `requirements.txt` includes `audioop-lts` which is needed on Python 3.13+.

**"ffmpeg not found"**
```bash
# macOS
brew install ffmpeg

# Windows (pick one)
winget install ffmpeg
choco install ffmpeg

# Linux
sudo apt install ffmpeg
```

**API key prompt every time**
```bash
# Just let the script save it to .env on first run (recommended)

# Or set it manually:
# macOS / Linux
export GEMINI_API_KEY="your_key"

# Windows (PowerShell)
$env:GEMINI_API_KEY="your_key"

# Windows (Command Prompt)
set GEMINI_API_KEY=your_key
```

**Rate limit errors** — Reduce `MAX_CONCURRENT_CHUNKS` to 3 or 2 in the script.

**Duplicate files in selection (Windows)** — Fixed in latest version. Update your script.

## Tips

- Process long files (>15 min) for best results with chunking
- Use `all` selection for batch processing overnight
- Converted files are auto-deleted, original files stay safe
- Output files are named `*_transcript_parallel.txt` (and `*_en.txt` for translations)
- The original transcript is always saved before translation starts

## Files

| File | Description |
|------|-------------|
| `transcribe_audio.py` | Main transcription script |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Automated setup for macOS / Linux |
| `setup.bat` | Automated setup for Windows |
| `.env` | API key storage (created on first run) |
| `.gitignore` | Excludes sensitive files from git |

## Get API Key

https://aistudio.google.com/app/apikey

- Free tier: 10 requests/minute
- Paid tier: 150+ requests/minute
