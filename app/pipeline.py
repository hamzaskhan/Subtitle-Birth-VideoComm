from pathlib import Path
import whisper
import ffmpeg
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
import google.generativeai as genai

# for dubbing
# from TTS.api import TTS   # Coqui XTTS v2 or OpenVoice
import tempfile

# Load API key from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=gemini_api_key)

try:
    _GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
    # Test the model with a simple prompt
    test_resp = _GEMINI_MODEL.generate_content("Hello")
    print("Gemini model initialized successfully")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    raise RuntimeError(f"Failed to initialize Gemini model: {e}")

# preload TTS model once
# XTTS v2 supports cross-lingual cloning with reference audio
# tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")


def _srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,ms)."""
    ms = int(float(seconds or 0) * 1000)
    return f"{ms//3600000:02d}:{(ms%3600000)//60000:02d}:{(ms%60000)//1000:02d},{ms%1000:03d}"


def _write_srt(segments, path: Path) -> None:
    """Write transcription segments to an SRT file."""
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(
                f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{seg['text'].strip()}\n\n"
            )


# Language mapping for better Gemini prompts
LANGUAGE_NAMES = {
    "en": "English",
    "ur": "Urdu", 
    "hi": "Hindi",
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese"
}

def transcribe_video(
    video_path: Path,
    output_dir: Path,
    model_size: str = "base",
    target_lang: str = "en",
) -> Tuple[List[Dict], Path]:
    """Transcribe audio and translate if needed. Returns (segments, srt_path)."""

    model = whisper.load_model(model_size)
    result = model.transcribe(str(video_path))
    segments = result["segments"]

    # Whisper transcribes in the original language (Japanese in this case)
    # We need to translate to the target language regardless of what it is
    # For English subtitles, we need to translate Japanese to English
    # For other languages, we need to translate Japanese to that language
    
    # Check if we have any text to translate
    if not segments or not any(seg["text"].strip() for seg in segments):
        print("No text found to translate")
        lang_suffix = target_lang
        srt_path = output_dir / f"{video_path.stem}.{lang_suffix}.srt"
        _write_srt(segments, srt_path)
        return segments, srt_path
    
    print(f"Translating Japanese audio to {target_lang}...")
    # Use Gemini to translate while preserving alignment.
    texts = [seg["text"] for seg in segments]
    
    # Store original text for debugging
    for seg in segments:
        seg["original_text"] = seg["text"]

    cleaned_all: list[str] = []
    chunk_size = 80  # translate max 80 lines per call to stay under token limit

    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        numbered = "\n".join(f"{idx+1}. {t}" for idx, t in enumerate(chunk))
        target_lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        prompt = (
            f"Translate the following subtitles into {target_lang_name}.\n"
            "Keep the original numbering, one line per subtitle, and output nothing else.\n"
            "Ensure the translation is natural and contextually appropriate.\n\n"
            f"{numbered}"
        )

        try:
            resp = _GEMINI_MODEL.generate_content(prompt)
            if not resp.text:
                raise RuntimeError("Empty response from Gemini")
            lines = [ln.strip() for ln in resp.text.split("\n") if ln.strip()]
            print(f"Gemini response for chunk {start//chunk_size + 1}: {len(lines)} lines")
        except Exception as e:
            print(f"Gemini API error: {e}")
            print(f"Prompt sent: {prompt}")
            # Try with a simpler prompt as fallback
            try:
                simple_prompt = f"Translate these Japanese subtitles to {target_lang_name}:\n{numbered}"
                resp = _GEMINI_MODEL.generate_content(simple_prompt)
                if resp.text:
                    lines = [ln.strip() for ln in resp.text.split("\n") if ln.strip()]
                    print(f"Fallback translation successful: {len(lines)} lines")
                else:
                    raise RuntimeError("Empty response from fallback translation")
            except Exception as fallback_e:
                print(f"Fallback translation also failed: {fallback_e}")
                raise RuntimeError(f"Gemini translation failed: {e}")

        # Strip leading numbers "N. "
        for ln in lines:
            if ln.split(" ")[0].rstrip(".").isdigit():
                cleaned_all.append(" ".join(ln.split(" ")[1:]))
            else:
                cleaned_all.append(ln)

    if len(cleaned_all) != len(segments):
        print(f"Warning: Translation line count mismatch: expected {len(segments)}, got {len(cleaned_all)}")
        print("Using original text for missing translations")
        # Pad with original text if translation is incomplete
        while len(cleaned_all) < len(segments):
            cleaned_all.append(segments[len(cleaned_all)]["text"])

    for seg, tr in zip(segments, cleaned_all):
        seg["text"] = tr
        print(f"Original: {seg.get('original_text', 'N/A')} -> Translated: {tr}")

    lang_suffix = target_lang
    srt_path = output_dir / f"{video_path.stem}.{lang_suffix}.srt"
    _write_srt(segments, srt_path)
    return segments, srt_path


def burn_subtitles(video_path: Path, srt_path: Path, output_dir: Path, target_lang: str = "en") -> Path:
    """Burn SRT into video and return path."""
    out_file = output_dir / f"{video_path.stem}.{target_lang}_burned.mp4"
    
    try:
        print(f"Starting subtitle burning for {target_lang}...")
        
        # More efficient approach with better encoding settings
        (
            ffmpeg.input(str(video_path))
            .output(
                str(out_file),
                vf=f"subtitles='{srt_path.as_posix()}':force_style='FontSize=24,PrimaryColour=&Hffffff&,OutlineColour=&H000000&,BorderStyle=3'",
                vcodec="libx264",
                acodec="copy",  # Copy audio instead of re-encoding - MUCH FASTER
                preset="ultrafast",  # Use ultrafast encoding preset
                crf=28,  # Slightly lower quality but much faster
                movflags="+faststart",
                loglevel="info"
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=False)
        )
        print(f"Subtitle burning completed: {out_file}")
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise RuntimeError(f"ffmpeg error:\n{e.stderr.decode() if e.stderr else str(e)}\n") from e
    return out_file


# def dub_audio(
#     video_path: Path, segments: List[Dict], output_dir: Path, target_lang: str = "en"
# ) -> Path:
#     """Generate dubbed audio using XTTS v2 and mux with video."""
#     audio_out = output_dir / f"{video_path.stem}.{target_lang}.dub.wav"
#     dubbed_video = output_dir / f"{video_path.stem}.{target_lang}.dub.mp4"

#     # extract original audio for voice reference
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref:
#         ffmpeg.input(str(video_path)).output(ref.name, acodec="pcm_s16le", ac=1, ar="16k").overwrite_output().run()
#         ref_audio = ref.name

#     # join translated lines into a single text (XTTS generates longer speech better than line-by-line)
#     joined_text = " ".join([seg["text"] for seg in segments])

#     # synthesize translated speech with voice cloning
#     tts_model.tts_to_file(
#         text=joined_text,
#         speaker_wav=ref_audio,   # cloning voice
#         language=target_lang,
#         file_path=str(audio_out),
#     )

#     # replace video audio
#     (
#         ffmpeg.input(str(video_path))
#         .output(
#             str(dubbed_video),
#             vcodec="copy",
#             acodec="aac",
#             audio_bitrate="192k",
#             map="0:v:0",
#             **{"i": str(audio_out)},
#         )
#         .overwrite_output()
#         .run(capture_stdout=True, capture_stderr=True)
#     )

#     return dubbed_video


def process_video(
    video_path: str | Path,
    output_dir: str | Path = "storage",
    mode: str = "sub",  # "sub" for subtitles, "dub" for dubbing
    target_lang: str = "en",
) -> Tuple[Path, str, List[Dict]]:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(video)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # transcription + optional translation
    segments, srt_path = transcribe_video(video, output_dir, target_lang=target_lang)
    transcript_text = " ".join(seg["text"] for seg in segments)

    if mode == "sub":
        out_file = burn_subtitles(video, srt_path, output_dir, target_lang)
    # elif mode == "dub":
    #     out_file = dub_audio(video, segments, output_dir, target_lang=target_lang)
    else:
        raise ValueError("mode must be 'sub' or 'dub'")

    return out_file, transcript_text, segments
