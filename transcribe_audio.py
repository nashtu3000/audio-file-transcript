#!/usr/bin/env python3
"""
Parallel Audio Transcription with Gemini 2.0 Flash
- Optimized for paid tier accounts (Tier 1: 150-300 RPM, Tier 2: 1000+ RPM)
- Processes multiple chunks concurrently for faster transcription
- Intelligent rate limiting and retry logic
- Progress tracking for parallel operations
"""

import os
import time
import re
import shutil
import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from google import genai
from google.genai import types

# Try to import pydub for audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError as e:
    PYDUB_AVAILABLE = False
    if "audioop" in str(e):
        print("Warning: pydub requires 'audioop-lts' on Python 3.13+")
        print("  Fix: pip install audioop-lts")
    else:
        print(f"Warning: pydub not installed. Install with: pip install pydub")
        print(f"  Error: {e}")

# Configuration for Paid Tier
CHUNK_DURATION_MINUTES = 10  # Slightly smaller chunks for better parallelization
OVERLAP_SECONDS = 3
MAX_RETRIES = 3
RETRY_DELAY = 2  # Shorter delay since we have higher quotas

# Parallel Processing Configuration
MAX_CONCURRENT_CHUNKS = 5  # Conservative limit for reliable processing
RATE_LIMIT_BUFFER = 0.5  # Small delay between chunk submissions to avoid bursts

# Media file extensions to search for (lowercase + uppercase for case-sensitive filesystems)
SUPPORTED_AUDIO_FORMATS = ['mp3', 'MP3', 'm4a', 'M4A', 'wav', 'WAV', 'aac', 'AAC',
                           'ogg', 'OGG', 'flac', 'FLAC', 'wma', 'WMA']
SUPPORTED_VIDEO_FORMATS = ['mp4', 'MP4', 'mov', 'MOV', 'avi', 'AVI', 'mkv', 'MKV',
                           'webm', 'WEBM', 'flv', 'FLV']

# MP3 conversion settings (balance between quality and cost)
MP3_BITRATE = "64k"  # Gemini downsamples to 16kbps, so 64k is more than enough
MP3_SAMPLE_RATE = "16000"  # 16kHz matches Gemini's processing

# Silence detection thresholds
SILENCE_DBFS_THRESHOLD = -45  # Below this = near-digital-silence, skip entirely
SILENCE_DETECT_THRESHOLD = -35  # dBFS threshold for speech vs silence boundaries
MIN_SPEECH_RATIO = 0.10  # At least 10% of chunk must contain speech
MIN_SILENCE_LEN_MS = 700  # 700ms of quiet counts as a silence gap

# Known hallucination phrases (LLMs commonly generate these on silent audio)
HALLUCINATION_PHRASES = {
    "subtitles by", "transcribed by", "thanks for watching",
    "thank you for watching", "please subscribe", "like and subscribe",
    "subtítulos", "amara.org", "translated by", "copyright",
    "all rights reserved", "no speech detected", "music playing",
    "silence", "[music]", "[applause]", "end of transcript",
}

# Thread-safe progress tracking
progress_lock = Lock()
completed_chunks = 0
total_chunks = 0

@dataclass
class TranscriptChunk:
    """Represents a transcribed chunk with metadata"""
    chunk_index: int
    start_time: float
    end_time: float
    text: str
    processing_time: float = 0.0

def get_api_key() -> str:
    """Get API key from .env file, environment variable, or prompt user"""
    # First, try to load from .env file in current directory
    env_file = Path.cwd() / '.env'
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('GEMINI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        if api_key:
                            return api_key
        except Exception as e:
            print(f"⚠️  Warning: Could not read .env file: {str(e)}")

    # Second, check environment variable
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        return api_key

    # If not found, prompt user
    print("\n" + "="*80)
    print("GEMINI API KEY REQUIRED")
    print("="*80)
    print("No API key found in .env file or GEMINI_API_KEY environment variable")
    print("\nGet your API key from: https://aistudio.google.com/app/apikey")
    print("="*80)

    api_key = input("\nPlease enter your Gemini API key: ").strip()

    if not api_key:
        print("❌ Error: No API key provided. Exiting.")
        exit(1)

    print("✓ API key accepted")

    # Offer to save to .env file
    save_choice = input("\n💾 Save API key to .env file for future use? (y/n): ").strip().lower()
    if save_choice == 'y':
        try:
            with open(env_file, 'w') as f:
                f.write(f'# Gemini API Configuration\n')
                f.write(f'# Get your API key from: https://aistudio.google.com/app/apikey\n')
                f.write(f'GEMINI_API_KEY={api_key}\n')
            print(f"✓ API key saved to {env_file}")
            print("  You won't be prompted again in this directory!")
        except Exception as e:
            print(f"⚠️  Warning: Could not save to .env file: {str(e)}")
            print("  You can manually create a .env file with:")
            print(f"  GEMINI_API_KEY={api_key}")

    return api_key

def format_duration(seconds: float) -> str:
    """Format duration as 'X seconds (Y minutes)'"""
    minutes = seconds / 60
    return f"{seconds:.1f}s ({minutes:.1f}min)"

def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def get_recording_date(file_path: Path) -> Optional[str]:
    """
    Extract the recording date from a media file.
    Returns date as 'yyyy-mm-dd' string, or None if not found.

    Priority order:
      1. ffprobe creation_time metadata tag (actual recording date)
      2. Date parsed from the filename (patterns like GMT20260212, 2026-02-12, 20260212)
      3. File modification time (last resort)
    """
    recording_date = None

    # --- Strategy 1: ffprobe metadata ---
    recording_date = _date_from_ffprobe(file_path)
    if recording_date:
        return recording_date

    # --- Strategy 2: Parse date from filename ---
    recording_date = _date_from_filename(file_path.name)
    if recording_date:
        return recording_date

    # --- Strategy 3: File modification time (fallback) ---
    try:
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
    except Exception:
        return date.today().isoformat()


def _date_from_ffprobe(file_path: Path) -> Optional[str]:
    """Extract creation_time from file metadata using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
             str(file_path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None

        metadata = json.loads(result.stdout)
        creation_time = metadata.get('format', {}).get('tags', {}).get('creation_time')
        if not creation_time:
            return None

        # Parse ISO format like "2026-02-12T12:31:02.000000Z"
        dt = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d')
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, FileNotFoundError):
        return None


def _date_from_filename(filename: str) -> Optional[str]:
    """
    Try to extract a date from the filename.
    Recognises common patterns:
      - GMT20260212-123102  (Zoom-style)
      - 20260212            (compact yyyymmdd)
      - 2026-02-12          (ISO yyyy-mm-dd)
    """
    # Pattern 1: GMT followed by yyyymmdd
    m = re.search(r'GMT(\d{4})(\d{2})(\d{2})', filename)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d.isoformat()
        except ValueError:
            pass

    # Pattern 2: ISO date yyyy-mm-dd already in the filename
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d.isoformat()
        except ValueError:
            pass

    # Pattern 3: Compact yyyymmdd (standalone, not part of a longer number)
    m = re.search(r'(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)', filename)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d.isoformat()
        except ValueError:
            pass

    return None


def _filename_already_has_date_prefix(filename: str) -> bool:
    """Check if filename already starts with yyyy-mm-dd."""
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}\s', filename))


def build_output_filename(original_file: Path, suffix: str = '') -> str:
    """
    Build the output filename with date prefix.
    - Extracts recording date from metadata / filename / mtime.
    - Only prepends date if filename doesn't already start with yyyy-mm-dd.
    - suffix: optional suffix like '_en' before .txt extension.
    """
    base_name = original_file.stem.replace('_converted', '')

    if _filename_already_has_date_prefix(base_name):
        # Already has a date prefix — use as-is
        return f"{base_name}{suffix}.txt"

    recording_date = get_recording_date(original_file)
    if recording_date:
        return f"{recording_date} {base_name}{suffix}.txt"
    else:
        return f"{base_name}{suffix}.txt"


def update_progress(increment: int = 1):
    """Thread-safe progress update"""
    global completed_chunks
    with progress_lock:
        completed_chunks += increment
        percentage = (completed_chunks / total_chunks) * 100
        print(f"  Progress: [{completed_chunks}/{total_chunks}] {percentage:.1f}% complete")

def split_audio_file(file_path: str, chunk_duration_ms: int, overlap_ms: int) -> List[Tuple[AudioSegment, float, float]]:
    """Split audio file into overlapping chunks"""
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub is required. Install with: pip install pydub")

    print(f"Loading audio file: {file_path}")
    audio = AudioSegment.from_mp3(file_path)
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000

    print(f"Audio duration: {format_duration(total_duration_sec)}")

    chunks = []
    start_ms = 0
    chunk_index = 0

    while start_ms < total_duration_ms:
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        chunk = audio[start_ms:end_ms]
        start_sec = start_ms / 1000
        end_sec = end_ms / 1000

        chunks.append((chunk, start_sec, end_sec))
        print(f"  Chunk {chunk_index + 1}: {format_timestamp(start_sec)} - {format_timestamp(end_sec)}")

        start_ms = end_ms - overlap_ms
        chunk_index += 1

        if end_ms >= total_duration_ms:
            break

    print(f"Split into {len(chunks)} chunks for parallel processing")
    return chunks

def save_audio_chunk(chunk: AudioSegment, output_path: str) -> str:
    """Save audio chunk to temporary file"""
    chunk.export(output_path, format="mp3")
    return output_path

def detect_speech_in_chunk(audio_chunk: AudioSegment) -> Tuple[bool, float, int]:
    """
    Detect whether an audio chunk contains actual speech.
    Returns (has_speech, speech_ratio, speech_start_ms).
    speech_start_ms = millisecond offset where the first speech begins.
    Uses two layers: dBFS gate + non-silent region detection.
    """
    from pydub.silence import detect_nonsilent

    # Layer 1: Overall volume check — reject near-digital-silence
    if audio_chunk.dBFS < SILENCE_DBFS_THRESHOLD:
        return False, 0.0, 0

    # Layer 2: Find non-silent regions and calculate speech ratio
    nonsilent_ranges = detect_nonsilent(
        audio_chunk,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_DETECT_THRESHOLD,
        seek_step=10  # 10ms steps (faster than default 1ms)
    )

    if not nonsilent_ranges:
        return False, 0.0, 0

    speech_ms = sum(end - start for start, end in nonsilent_ranges)
    speech_ratio = speech_ms / len(audio_chunk)
    speech_start_ms = nonsilent_ranges[0][0]

    return speech_ratio >= MIN_SPEECH_RATIO, speech_ratio, speech_start_ms

def count_valid_transcript_lines(text: str) -> int:
    """Count lines that match the expected transcript format [MM:SS] Speaker N: ..."""
    count = 0
    for line in text.strip().split('\n'):
        if re.match(r'^\[?\d{1,2}:\d{2}\]\s*Speaker\s+\d+', line.strip()):
            count += 1
    return count

def clean_transcript_output(raw_text: str) -> str:
    """
    Remove non-transcript lines (commentary, hallucinated annotations) but
    keep valid transcript lines.  Works line-by-line so a single bad line
    does not discard an entire chunk.
    """
    lines = raw_text.strip().split('\n')
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Only keep lines that start with a timestamp [MM:SS]
        if not re.match(r'^\[?\d{1,2}:\d{2}\]', stripped):
            continue  # skip commentary / headers / hallucinated prose

        # Check this individual line against hallucination phrases
        line_lower = stripped.lower()
        is_bad = any(phrase in line_lower for phrase in HALLUCINATION_PHRASES)
        if not is_bad:
            cleaned.append(line)

    return '\n'.join(cleaned)

def is_hallucinated_output(text: str) -> bool:
    """Check if transcription output is likely hallucinated."""
    text_lower = text.strip().lower()

    # Empty or very short output
    if len(text_lower) < 10:
        return True

    # No timestamp brackets at all → not a valid transcript
    if '[' not in text or ']' not in text:
        return True

    # Count properly-formatted transcript lines.
    # If there are several valid lines, the chunk contains real speech
    # regardless of whether a hallucination phrase also appears somewhere.
    valid_lines = count_valid_transcript_lines(text)
    if valid_lines >= 3:
        return False

    # Very few valid lines — check for hallucination phrases
    for phrase in HALLUCINATION_PHRASES:
        if phrase in text_lower:
            return True

    return False

def transcribe_chunk_worker(
    chunk_data: Tuple[int, str, float, str, str]
) -> Optional[TranscriptChunk]:
    """
    Worker function for parallel chunk transcription
    Args: (chunk_index, chunk_path, start_time, original_filename, api_key)
    """
    chunk_index, chunk_path, start_time, original_filename, api_key = chunk_data

    start_processing = time.time()

    # --- SILENCE PRE-CHECK: skip API call for silent chunks ---
    trim_offset_sec = 0.0  # extra offset if we trim leading silence
    try:
        chunk_audio = AudioSegment.from_mp3(chunk_path)
        has_speech, speech_ratio, speech_start_ms = detect_speech_in_chunk(chunk_audio)
        if not has_speech:
            print(f"  ⏭️  Chunk {chunk_index + 1}: SKIPPED (silence detected, speech ratio: {speech_ratio:.1%})")
            update_progress()
            return TranscriptChunk(
                chunk_index=chunk_index,
                start_time=start_time,
                end_time=start_time + 600,
                text="",  # Empty text, will be filtered during merge
                processing_time=time.time() - start_processing
            )
        print(f"  🔊 Chunk {chunk_index + 1}: Speech detected ({speech_ratio:.0%} of audio)")

        # Trim leading silence if speech starts late (>5s into the chunk).
        # This prevents Gemini from hallucinating on long silent intros.
        if speech_start_ms > 5000:
            trim_start_ms = max(0, speech_start_ms - 2000)  # keep 2s buffer
            chunk_audio = chunk_audio[trim_start_ms:]
            chunk_audio.export(chunk_path, format="mp3")
            trim_offset_sec = trim_start_ms / 1000.0
            print(f"  ✂️  Chunk {chunk_index + 1}: Trimmed {trim_offset_sec:.1f}s of leading silence")
    except Exception as e:
        # If silence detection fails, proceed with transcription anyway
        print(f"  ⚠️  Chunk {chunk_index + 1}: Could not check for silence ({e}), proceeding...")

    client = genai.Client(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  ⏳ Chunk {chunk_index + 1}: Uploading to Gemini...")

            # Upload and process
            upload_config = types.UploadFileConfig(mime_type="audio/mpeg")
            audio_file = client.files.upload(file=chunk_path, config=upload_config)

            print(f"  ✓ Chunk {chunk_index + 1}: Upload complete, waiting for processing...")

            # Wait for processing
            while audio_file.state == "PROCESSING":
                time.sleep(1)
                audio_file = client.files.get(name=audio_file.name)

            if audio_file.state == "FAILED":
                raise Exception("File processing failed")

            print(f"  🔄 Chunk {chunk_index + 1}: Starting transcription with Gemini...")

            # Transcription prompt — timestamps must be RELATIVE to audio start (00:00)
            # We handle the absolute offset later via adjust_timestamps()
            prompt = f"""Analyze this audio segment and provide a complete verbatim transcript with speaker diarization.

Audio file: {original_filename} (Segment {chunk_index + 1})

SILENCE AND HALLUCINATION PREVENTION (HIGHEST PRIORITY):
- If the audio contains NO intelligible human speech — only silence, background noise, music, or unintelligible sounds — respond with EXACTLY: [NO_SPEECH_DETECTED]
- Do NOT invent, fabricate, or guess at words that are not clearly spoken.
- Do NOT generate filler text such as "Subtitles by...", "Thanks for watching", "Transcribed by...", or any metadata.
- Do NOT describe sounds (e.g., "[music playing]", "[silence]") — only transcribe actual spoken words.
- If you are uncertain whether something is speech, OMIT it. Silence is always preferable to fabrication.
- ONLY transcribe words that are ACTUALLY and CLEARLY SPOKEN by a human voice.

SPEAKER DIARIZATION:
- Pay EXTREME attention to voice characteristics (pitch, tone, gender, accent, style).
- When speaker changes, create NEW timestamp entry.
- Listen for conversational turn-taking.
- Questions vs. Answers often indicate different speakers.
- Long monologues = ONE speaker. Short responses = often DIFFERENT speaker.
- CONSISTENCY: Use "Speaker 1" and "Speaker 2" labels ONLY. Do NOT use "Female Speaker", "Male Speaker", "Speaker A", "Speaker B" or any other variations.
- If you detect a third speaker, use "Speaker 3", fourth use "Speaker 4", etc.
- MAINTAIN these same labels throughout the ENTIRE segment - do not switch naming conventions.

Instructions:
1. Transcribe ONLY what is actually said, word-for-word
2. Label speakers as "Speaker 1", "Speaker 2", "Speaker 3", etc. based on when they first appear
3. NEVER change speaker labels mid-transcript (if you called someone "Speaker 1", keep calling them "Speaker 1")
4. Include timestamps for EVERY speaker change (format: [MM:SS])
5. Timestamps must be RELATIVE to the start of this audio clip (start at [00:00])
6. Preserve original language
7. Group consecutive utterances by SAME speaker

Output format:
[MM:SS] Speaker 1: Text of what they said.
[MM:SS] Speaker 2: Response from second speaker.

If no speech: [NO_SPEECH_DETECTED]"""

            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_uri(
                                file_uri=audio_file.uri,
                                mime_type=audio_file.mime_type
                            ),
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                )
            )

            # Cleanup uploaded file
            client.files.delete(name=audio_file.name)

            raw_text = response.text.strip()

            # --- HALLUCINATION POST-CHECK ---
            # Explicit no-speech marker from the model
            if "[NO_SPEECH_DETECTED]" in raw_text:
                print(f"  ⏭️  Chunk {chunk_index + 1}: No speech detected by model")
                update_progress()
                return TranscriptChunk(
                    chunk_index=chunk_index,
                    start_time=start_time,
                    end_time=start_time + 600,
                    text="",
                    processing_time=time.time() - start_processing
                )

            # Clean output: strip commentary/hallucinated lines, keep valid transcript lines
            cleaned_text = clean_transcript_output(raw_text)

            if not cleaned_text.strip() or is_hallucinated_output(cleaned_text):
                print(f"  ⏭️  Chunk {chunk_index + 1}: No valid transcript / hallucination filtered")
                update_progress()
                return TranscriptChunk(
                    chunk_index=chunk_index,
                    start_time=start_time,
                    end_time=start_time + 600,
                    text="",
                    processing_time=time.time() - start_processing
                )

            valid_count = count_valid_transcript_lines(cleaned_text)
            print(f"  📝 Chunk {chunk_index + 1}: {valid_count} valid transcript lines after cleaning")

            # Adjust timestamps: add chunk start offset + any trim offset
            total_offset = start_time + trim_offset_sec
            transcript = adjust_timestamps(cleaned_text, total_offset)

            processing_time = time.time() - start_processing

            # Update progress
            update_progress()

            return TranscriptChunk(
                chunk_index=chunk_index,
                start_time=start_time,
                end_time=start_time + 600,  # Approximate
                text=transcript,
                processing_time=processing_time
            )

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"  Chunk {chunk_index + 1} error (attempt {attempt + 1}): {str(e)}")
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Chunk {chunk_index + 1} failed after {MAX_RETRIES} attempts: {str(e)}")
                update_progress()  # Count as completed even if failed
                return None

    return None

def adjust_timestamps(transcript: str, offset_seconds: float) -> str:
    """Adjust timestamps in transcript by adding offset"""
    def replace_timestamp(match):
        time_str = match.group(1)
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = int(parts[1])

        total_seconds = minutes * 60 + seconds + offset_seconds
        new_minutes = int(total_seconds // 60)
        new_seconds = int(total_seconds % 60)

        return f"[{new_minutes:02d}:{new_seconds:02d}]"

    pattern = r'\[(\d{2}:\d{2})\]'
    return re.sub(pattern, replace_timestamp, transcript)

def merge_transcripts(chunks: List[TranscriptChunk]) -> str:
    """Merge transcript chunks, filtering out empty/silent ones"""
    if not chunks:
        return ""

    # Sort by chunk index to ensure proper order
    sorted_chunks = sorted(chunks, key=lambda x: x.chunk_index)

    # Filter out empty chunks (silence / hallucination-filtered)
    texts = [chunk.text for chunk in sorted_chunks if chunk.text.strip()]

    return "\n\n".join(texts)

def consolidate_speakers(transcript: str, api_key: str) -> str:
    """
    Use AI to consolidate duplicate speaker labels in merged transcript.
    E.g., "Female Speaker" and "Speaker A" might be the same person.
    """
    if not transcript or "[NO_SPEECH_DETECTED]" in transcript:
        return transcript

    print(f"🔄 Analyzing speaker consistency...")

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""Consolidate duplicate speaker labels in this transcript.

The transcript was generated in chunks, so the same speaker may have different labels like:
- "Speaker 1" and "Speaker A" (same person)
- "Female Speaker" and "Speaker 1" (same person)
- "Male Speaker" and "Speaker 2" (same person)

CRITICAL INSTRUCTIONS:
1. Identify which speaker labels refer to the same person based on context and conversation flow
2. Standardize ALL speaker labels to use only: "Speaker 1", "Speaker 2", "Speaker 3", etc.
3. Maintain chronological order - the first speaker to talk is "Speaker 1", second is "Speaker 2", etc.
4. Keep ALL the content exactly as is - only change the speaker labels
5. Preserve timestamps, text, and formatting exactly
6. DO NOT add any commentary, analysis, or explanation
7. DO NOT add introductory text or conclusions
8. Return ONLY the transcript lines starting with timestamps [MM:SS]

TRANSCRIPT TO CONSOLIDATE:
{transcript}

OUTPUT FORMAT - Return ONLY transcript lines like this (no commentary):
[00:00] Speaker 1: Text here.
[00:05] Speaker 2: Response here."""

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )

        consolidated = response.text.strip()

        # Remove any commentary - keep only lines starting with timestamps
        lines = consolidated.split('\n')
        transcript_lines = []
        for line in lines:
            # Keep lines that start with timestamp [MM:SS] or are empty (for spacing)
            if line.strip().startswith('[') or line.strip() == '':
                transcript_lines.append(line)

        # Join back and clean up
        cleaned_transcript = '\n'.join(transcript_lines).strip()

        # Safety check: consolidation must not silently drop significant content
        original_line_count = len([l for l in transcript.split('\n') if l.strip().startswith('[')])
        consolidated_line_count = len([l for l in cleaned_transcript.split('\n') if l.strip().startswith('[')])

        if consolidated_line_count < original_line_count * 0.8:
            lost = original_line_count - consolidated_line_count
            print(f"⚠️  Speaker consolidation lost {lost}/{original_line_count} lines — keeping original transcript")
            return transcript

        print(f"✓ Speaker labels consolidated ({consolidated_line_count} lines)")
        return cleaned_transcript

    except Exception as e:
        print(f"⚠️  Could not consolidate speakers: {str(e)}")
        print(f"   Continuing with original transcript")
        return transcript

def translate_transcript_to_english(transcript: str, api_key: str) -> Optional[str]:
    """Translate transcript to English using Gemini, preserving format exactly."""
    if not transcript or "[NO_SPEECH_DETECTED]" in transcript:
        return transcript

    print(f"\n🌐 Translating transcript to English...")

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""Translate this transcript to English.

CRITICAL RULES:
1. This is a LINE-BY-LINE, 1-to-1 translation. Each line in the input produces exactly one line in the output.
2. Keep ALL timestamps [MM:SS] exactly as they are - do not change them.
3. Keep ALL speaker labels (Speaker 1, Speaker 2, etc.) exactly as they are.
4. ONLY translate the spoken text that comes after "Speaker N: ".
5. Preserve the exact format: [MM:SS] Speaker N: translated text
6. Do NOT add, remove, merge, or split any lines.
7. Do NOT add any commentary, notes, headers, or explanations.
8. Output ONLY the translated transcript lines, nothing else.

TRANSCRIPT TO TRANSLATE:
{transcript}"""

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )

        translated = response.text.strip()

        # Clean: keep only timestamp lines
        lines = translated.split('\n')
        transcript_lines = []
        for line in lines:
            if line.strip().startswith('[') or line.strip() == '':
                transcript_lines.append(line)

        cleaned = '\n'.join(transcript_lines).strip()
        print(f"✓ Translation complete")
        return cleaned

    except Exception as e:
        print(f"⚠️  Translation failed: {str(e)}")
        return None


def count_speakers(transcript_text: str) -> Tuple[int, List[str]]:
    """Count unique speakers in transcript"""
    speakers = set()
    for line in transcript_text.split('\n'):
        if ':' in line and line.strip():
            parts = line.split(']', 1)
            if len(parts) > 1:
                speaker_part = parts[1].split(':', 1)
                if len(speaker_part) > 1:
                    speaker = speaker_part[0].strip()
                    if speaker:
                        speakers.add(speaker)
    return len(speakers), sorted(list(speakers))

def format_transcript(filename: str, transcript_text: str, duration: float, processing_stats: dict) -> str:
    """Format transcript with header"""
    num_speakers, speaker_list = count_speakers(transcript_text)
    speakers_str = ", ".join(speaker_list) if speaker_list else "Unknown"

    output = "=" * 80 + "\n"
    output += "MEETING TRANSCRIPT (Gemini Parallel Processing)\n"
    output += "=" * 80 + "\n"
    output += f"Source File: {filename}\n"
    output += f"Duration: {format_duration(duration)}\n"
    output += f"Speakers: {num_speakers}: {speakers_str}\n"
    output += f"Processing: {CHUNK_DURATION_MINUTES}-min chunks, {MAX_CONCURRENT_CHUNKS} parallel workers\n"
    output += f"Total Processing Time: {format_duration(processing_stats['total_time'])}\n"
    output += f"Average Chunk Time: {format_duration(processing_stats['avg_chunk_time'])}\n"
    output += f"Speedup vs Sequential: {processing_stats['speedup_factor']:.1f}x\n"
    output += "=" * 80 + "\n\n"
    output += transcript_text

    return output

def process_audio_file_parallel(file_path: str) -> Optional[Tuple[str, str, str]]:
    """Process audio file with parallel chunking. Returns (formatted_output, raw_transcript, api_key)."""
    global completed_chunks, total_chunks

    print(f"\n{'='*80}")
    print(f"Processing: {file_path}")
    print(f"Parallel Processing Mode: {MAX_CONCURRENT_CHUNKS} concurrent workers")
    print(f"{'='*80}")

    if not PYDUB_AVAILABLE:
        print("Error: pydub not installed")
        return None

    try:
        overall_start = time.time()

        # Split audio
        chunk_duration_ms = CHUNK_DURATION_MINUTES * 60 * 1000
        overlap_ms = OVERLAP_SECONDS * 1000
        audio_chunks = split_audio_file(file_path, chunk_duration_ms, overlap_ms)

        # Initialize progress tracking
        completed_chunks = 0
        total_chunks = len(audio_chunks)

        # Get API key once (before spawning workers)
        print("\n🔑 Checking API key...")
        api_key = get_api_key()
        print("✓ API key validated\n")

        # Prepare temporary directory (clean up any leftovers first)
        temp_dir = Path(file_path).parent / "temp_chunks"
        if temp_dir.exists():
            print(f"🧹 Cleaning up leftover temp directory...")
            shutil.rmtree(temp_dir)
            print(f"✓ Cleanup complete")
        temp_dir.mkdir(exist_ok=True)

        # Save all chunks to temp files
        print(f"📦 Preparing {len(audio_chunks)} chunks for parallel processing...")
        chunk_tasks = []

        for i, (chunk_audio, start_time, end_time) in enumerate(audio_chunks):
            temp_chunk_path = temp_dir / f"chunk_{i}.mp3"
            save_audio_chunk(chunk_audio, str(temp_chunk_path))
            chunk_tasks.append((i, str(temp_chunk_path), start_time, Path(file_path).name, api_key))
            print(f"  ✓ Saved chunk {i + 1}/{len(audio_chunks)}")

        print(f"\n✓ All chunks prepared and saved to temporary files")
        print(f"\n{'='*80}")
        print(f"🚀 STARTING PARALLEL TRANSCRIPTION")
        print(f"{'='*80}")
        print(f"Workers: {MAX_CONCURRENT_CHUNKS} concurrent")
        print(f"Chunks: {len(audio_chunks)} total")
        print(f"Watch below for real-time progress from each worker...")
        print(f"{'='*80}\n")

        # Process chunks in parallel
        transcript_chunks = []
        chunk_times = []

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHUNKS) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(transcribe_chunk_worker, task): task[0]
                for task in chunk_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        transcript_chunks.append(result)
                        chunk_times.append(result.processing_time)
                        print(f"  ✅ Chunk {result.chunk_index + 1} COMPLETE in {format_duration(result.processing_time)}")
                except Exception as e:
                    print(f"  ❌ Chunk {chunk_idx + 1} FAILED: {str(e)}")

        print(f"\n{'='*80}")

        if not transcript_chunks:
            print("❌ Error: No chunks were successfully transcribed")
            # Cleanup temp directory before returning
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️  Could not clean up temp directory: {str(e)}")
            return None

        # Calculate statistics
        overall_time = time.time() - overall_start
        avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
        sequential_estimate = avg_chunk_time * len(audio_chunks)
        speedup_factor = sequential_estimate / overall_time if overall_time > 0 else 1

        print(f"\n{'='*80}")
        print(f"📊 PROCESSING STATISTICS")
        print(f"{'='*80}")
        print(f"✓ All chunks transcribed successfully!")
        print(f"  Chunks completed: {len(transcript_chunks)}/{len(audio_chunks)}")
        print(f"  Total time: {format_duration(overall_time)}")
        print(f"  Estimated sequential time: {format_duration(sequential_estimate)}")
        print(f"  ⚡ Speedup: {speedup_factor:.1f}x faster with parallel processing")
        print(f"{'='*80}")

        # Merge transcripts
        print(f"\n🔗 Merging {len(transcript_chunks)} transcript chunks...")
        merged_transcript = merge_transcripts(transcript_chunks)
        print(f"✓ Transcripts merged successfully")

        # Consolidate speaker labels for consistency
        merged_transcript = consolidate_speakers(merged_transcript, api_key)

        # Get total duration
        total_duration = audio_chunks[-1][2] if audio_chunks else 0

        # Format output
        processing_stats = {
            'total_time': overall_time,
            'avg_chunk_time': avg_chunk_time,
            'speedup_factor': speedup_factor
        }

        formatted_output = format_transcript(
            Path(file_path).name,
            merged_transcript,
            total_duration,
            processing_stats
        )

        # Cleanup temp directory (after successful processing)
        print(f"\n🧹 Cleaning up temporary chunk files...")
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"✓ Cleanup complete - temp directory removed")
        except Exception as cleanup_error:
            print(f"⚠️  Warning: Could not clean up temp directory: {str(cleanup_error)}")
            print(f"   You may need to manually delete: {temp_dir}")

        return (formatted_output, merged_transcript, api_key)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

        # Try to cleanup temp directory even on failure
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"✓ Temp directory cleaned up despite error")
        except:
            pass  # Silently ignore cleanup errors when already handling an exception

        return None

def find_all_media_files(directory: Path) -> List[Path]:
    """Find all supported media files in directory"""
    # Use dict keyed by lowercase name to deduplicate
    # (on Windows/macOS, *.mp3 and *.MP3 match the same files)
    seen = {}

    for ext in SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS:
        for f in directory.glob(f"*.{ext}"):
            seen[f.name.lower()] = f

    return sorted(seen.values(), key=lambda x: x.name.lower())

def convert_to_mp3(input_file: Path, output_file: Path) -> bool:
    """Convert media file to MP3 with optimal settings for transcription"""
    try:
        print(f"  Converting {input_file.name} to MP3...")

        audio = AudioSegment.from_file(str(input_file))

        # Export with optimal settings for Gemini transcription
        audio.export(
            str(output_file),
            format="mp3",
            bitrate=MP3_BITRATE,
            parameters=["-ar", MP3_SAMPLE_RATE, "-ac", "1"]  # mono, 16kHz
        )

        original_size = input_file.stat().st_size / (1024 * 1024)
        new_size = output_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Converted: {original_size:.1f}MB → {new_size:.1f}MB (saved {original_size - new_size:.1f}MB)")

        return True
    except Exception as e:
        print(f"  ✗ Conversion failed: {str(e)}")
        return False

def select_files_interactively(available_files: List[Path]) -> List[Path]:
    """Allow user to interactively select which files to process"""
    print("\n" + "="*80)
    print("MEDIA FILES FOUND")
    print("="*80)

    # Display files with numbers, format, and conversion status
    for idx, file in enumerate(available_files, 1):
        file_size = file.stat().st_size / (1024 * 1024)  # MB
        file_ext = file.suffix.upper()[1:]  # Remove dot

        # Check if conversion needed
        needs_conversion = file.suffix.lower() not in ['.mp3']
        conversion_note = " [will convert to MP3]" if needs_conversion else ""

        print(f"  [{idx}] {file.name:<50} {file_ext:>6} {file_size:>6.1f}MB{conversion_note}")

    print("="*80)
    print("\nSelect files to transcribe:")
    print("  - Enter numbers separated by commas (e.g., 1,3,5)")
    print("  - Enter 'all' to process all files")
    print("  - Enter 'q' to quit")
    print("="*80)

    while True:
        selection = input("\nYour selection: ").strip().lower()

        if selection == 'q':
            print("Exiting...")
            exit(0)

        if selection == 'all':
            print(f"✓ Selected all {len(available_files)} files")
            return available_files

        try:
            # Parse comma-separated numbers
            indices = [int(x.strip()) for x in selection.split(',')]

            # Validate indices
            if all(1 <= idx <= len(available_files) for idx in indices):
                selected_files = [available_files[idx - 1] for idx in indices]
                print(f"✓ Selected {len(selected_files)} file(s):")
                for f in selected_files:
                    print(f"    - {f.name}")
                return selected_files
            else:
                print(f"❌ Error: Please enter numbers between 1 and {len(available_files)}")
        except ValueError:
            print("❌ Error: Invalid input. Enter numbers separated by commas, 'all', or 'q'")

def select_output_language() -> str:
    """Allow user to select output language format"""
    print("\n" + "="*80)
    print("OUTPUT LANGUAGE")
    print("="*80)
    print("  [1] Original (keep original language)")
    print("  [2] Original with English translation")
    print("="*80)

    while True:
        selection = input("\nYour selection: ").strip()
        if selection == '1':
            return 'original'
        elif selection == '2':
            return 'translated'
        else:
            print("❌ Please enter 1 or 2")


def process_all_audio_files(specific_file: Optional[str] = None, interactive: bool = False):
    """Process all media files or a specific file"""
    current_dir = Path.cwd()

    if specific_file:
        file_path = current_dir / specific_file
        if not file_path.exists():
            print(f"Error: File '{specific_file}' not found")
            return
        selected_files = [file_path]
    else:
        all_media_files = find_all_media_files(current_dir)

        if not all_media_files:
            print("No media files found in current directory")
            print(f"Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS)}")
            return

        # Interactive selection if enabled
        if interactive:
            selected_files = select_files_interactively(all_media_files)
        else:
            selected_files = all_media_files

    if not selected_files:
        print("No files selected")
        return

    # Select output language
    language_choice = select_output_language()

    # Step 1: Extract recording dates from metadata BEFORE any processing
    print(f"\n{'='*80}")
    print("📅 READING FILE METADATA")
    print(f"{'='*80}")

    file_dates = {}  # original_file -> recording date string
    for file in selected_files:
        rec_date = get_recording_date(file)
        has_prefix = _filename_already_has_date_prefix(file.name)
        file_dates[file] = rec_date

        source = _date_from_ffprobe(file)
        if source:
            date_source = "metadata"
        elif _date_from_filename(file.name):
            date_source = "filename"
        else:
            date_source = "file modification time"

        if has_prefix:
            print(f"  📄 {file.name}")
            print(f"     Recording date: {rec_date} (from {date_source}) — already in filename")
        else:
            print(f"  📄 {file.name}")
            print(f"     Recording date: {rec_date} (from {date_source})")

    # Step 2: Convert non-MP3 files to MP3
    files_to_process = []   # (process_file, original_file) tuples
    converted_files = []    # Track for cleanup

    print(f"\n{'='*80}")
    print("PREPARING FILES")
    print(f"{'='*80}")

    for file in selected_files:
        if file.suffix.lower() == '.mp3':
            files_to_process.append((file, file))
            print(f"✓ {file.name} - already MP3, no conversion needed")
        else:
            # Convert to MP3
            temp_mp3_file = current_dir / f"{file.stem}_converted.mp3"
            if convert_to_mp3(file, temp_mp3_file):
                files_to_process.append((temp_mp3_file, file))  # keep ref to original
                converted_files.append(temp_mp3_file)
            else:
                print(f"⚠️  Skipping {file.name} due to conversion failure")

    if not files_to_process:
        print("No files to process after conversion")
        return

    print(f"\n{'='*80}")
    print(f"📝 TRANSCRIPTION QUEUE")
    print(f"{'='*80}")
    print(f"Files to process: {len(files_to_process)}")
    print(f"Parallel workers: {MAX_CONCURRENT_CHUNKS}")
    print(f"{'='*80}")

    try:
        for idx, (audio_file, original_file) in enumerate(files_to_process, 1):
            print(f"\n{'='*80}")
            print(f"📄 FILE {idx}/{len(files_to_process)}: {original_file.name}")
            print(f"{'='*80}")

            result = process_audio_file_parallel(str(audio_file))

            if result:
                formatted_output, raw_transcript, api_key = result

                # Build output filename with recording date from metadata
                output_filename = build_output_filename(original_file)
                output_path = current_dir / output_filename

                # Save original transcript immediately
                print(f"\n💾 Saving transcript...")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
                print(f"✅ Transcript saved to: {output_filename}")

                # Translate and save as separate file
                if language_choice == 'translated':
                    translated_text = translate_transcript_to_english(raw_transcript, api_key)
                    if translated_text:
                        en_filename = build_output_filename(original_file, suffix='_en')
                        en_path = current_dir / en_filename

                        # Reuse the header, swap in translated text
                        header_end = formatted_output.rfind("=" * 80 + "\n\n") + len("=" * 80 + "\n\n")
                        en_output = formatted_output[:header_end] + translated_text

                        print(f"💾 Saving English translation...")
                        with open(en_path, 'w', encoding='utf-8') as f:
                            f.write(en_output)
                        print(f"✅ English translation saved to: {en_filename}")
                    else:
                        print(f"⚠️  Translation failed, original transcript still saved")
            else:
                print(f"\n❌ FAILED to transcribe: {audio_file.name}")

            if idx < len(files_to_process):
                print(f"\n⏳ Waiting 5 seconds before next file...")
                time.sleep(5)

    finally:
        # Cleanup converted files
        if converted_files:
            print(f"\n{'='*80}")
            print("🧹 FINAL CLEANUP")
            print(f"{'='*80}")
            for temp_file in converted_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                        print(f"✓ Removed temporary file: {temp_file.name}")
                except Exception as e:
                    print(f"⚠️  Could not remove {temp_file.name}: {str(e)}")

    print(f"\n{'='*80}")
    print("🎉 ALL PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"✓ Processed {len(files_to_process)} file(s) successfully")
    print(f"✓ Check your directory for '*_transcript_parallel.txt' files")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    import sys

    if not PYDUB_AVAILABLE:
        print("\n" + "="*80)
        print("SETUP REQUIRED")
        print("="*80)
        print("Install dependencies: pip install -r requirements.txt")
        print("")
        print("Install ffmpeg:")
        print("  macOS:   brew install ffmpeg")
        print("  Windows: winget install ffmpeg")
        print("  Linux:   sudo apt install ffmpeg")
        print("="*80 + "\n")
        sys.exit(1)

    specific_file = sys.argv[1] if len(sys.argv) > 1 else None
    interactive = (specific_file is None)  # Enable interactive mode when no file specified
    process_all_audio_files(specific_file, interactive=interactive)
