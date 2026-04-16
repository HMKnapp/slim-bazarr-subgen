subgen_version = '26.04.14'

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

import xml.etree.ElementTree as ET
import threading
from threading import Lock
import sys
import time
import queue
import logging
import gc
import random
import re
import json
import urllib.request
import urllib.parse
from collections import Counter
from typing import Union
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse
import numpy as np

import stable_whisper
import faster_whisper

from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

def get_key_by_value(d, value):
    reverse_dict = {v: k for k, v in d.items()}
    return reverse_dict.get(value)

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')

def update_env_variables():
    global whisper_model, whisper_threads
    global concurrent_transcriptions, transcribe_device
    global webhookport, word_level_highlight, debug
    global model_location
    global transcribe_or_translate, force_detected_language_to
    global compute_type, reload_script_on_change
    global custom_model_prompt, custom_regroup
    global detect_language_length
    global tmdb_api_key

    whisper_model = os.getenv('WHISPER_MODEL', 'medium')
    whisper_threads = max(1, os.cpu_count() - 1)
    concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 1))
    transcribe_device = os.getenv('TRANSCRIBE_DEVICE', 'gpu')
    webhookport = int(os.getenv('WEBHOOKPORT', 9000))
    word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
    debug = convert_to_bool(os.getenv('DEBUG', False))
    model_location = '/models'
    transcribe_or_translate = os.getenv('TRANSCRIBE_OR_TRANSLATE', 'transcribe')
    force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', '').lower()
    # float32 matches original OpenAI Whisper precision without PyTorch's CUDA
    # library overhead. Override with COMPUTE_TYPE env var if needed.
    compute_type = os.getenv('COMPUTE_TYPE', 'float32')
    reload_script_on_change = convert_to_bool(os.getenv('RELOAD_SCRIPT_ON_CHANGE', False))
    custom_model_prompt = os.getenv('CUSTOM_MODEL_PROMPT', '')
    custom_regroup = os.getenv('CUSTOM_REGROUP', '')
    detect_language_length = os.getenv('DETECT_LANGUAGE_LENGTH', 240)
    # Override with TMDB_API_KEY env var to use your own key, or set it to
    # an empty string to disable prompt enrichment entirely.
    tmdb_api_key = os.getenv('TMDB_API_KEY', '')
   
    if transcribe_device == "gpu":
        transcribe_device = "cuda"

update_env_variables()

# ── quality/source tags that mark the end of the meaningful filename title ──
_QUALITY_RE = re.compile(
    r'[.\s](?:2160p|1080p|720p|480p|4k|bluray|blu-ray|bdrip|brrip|web[-.]?dl|webrip|'
    r'hmax|dsnp|amzn|nf|hulu|pcok|atvp|hdtv|dvdrip|remux|'
    r'av1|h\.?264|h\.?265|x264|x265|xvid|hevc|avc|'
    r'dolby|dts|truehd|atmos|aac|ac3|ddp|dd|flac|mp3|'
    r'hdr|hdr10|dv|sdr|imax|proper|repack|extended)',
    re.IGNORECASE,
)

# ── SxxExx TV episode pattern — optional year between title and SxxExx ──
_TV_RE = re.compile(
    r'^(.+?)[.\s](?:((?:19|20)\d{2})[.\s])?S(\d{1,2})E(\d{1,2})(?:[.\s](.+?))?(?=' + _QUALITY_RE.pattern + r'|$)',
    re.IGNORECASE,
)

# ── Movie year pattern — supports dot/space separator and (year) parentheses ──
_MOVIE_RE = re.compile(
    r'^(.+?)[.\s]\(?((?:19|20)\d{2})\)?(?=[.\s)]|$)',
    re.IGNORECASE,
)


def parse_video_filename(video_file: str) -> dict | None:
    """Return parsed metadata dict from a video filename, or None if unparseable."""
    name = os.path.splitext(os.path.basename(video_file))[0]

    tv = _TV_RE.match(name)
    if tv:
        show = tv.group(1).replace('.', ' ').strip()
        year = int(tv.group(2)) if tv.group(2) else None
        season = int(tv.group(3))
        episode = int(tv.group(4))
        ep_title_raw = tv.group(5)
        ep_title = ep_title_raw.replace('.', ' ').strip() if ep_title_raw else None
        # Drop ep_title if it looks like a quality tag
        if ep_title and _QUALITY_RE.search('.' + ep_title):
            ep_title = None
        return {'type': 'tv', 'show': show, 'year': year, 'season': season, 'episode': episode, 'ep_title': ep_title}

    movie = _MOVIE_RE.match(name)
    if movie:
        title = movie.group(1).replace('.', ' ').strip()
        year = int(movie.group(2))
        return {'type': 'movie', 'title': title, 'year': year}

    return None


_SELF_LABELS = {'self', 'himself', 'herself', 'themselves', 'narrator', 'host', 'various'}

def _cast_name(entry: dict) -> str | None:
    """Return the best name to use for a cast entry as a Whisper hotword.

    For fictional roles use the character name (what's spoken in dialogue).
    For documentaries/reality shows TMDB sets character to 'Self', 'Himself', etc.;
    in that case fall back to the real person's name since that's what's spoken.
    """
    character = (entry.get('character') or '').strip()
    if character and character.lower() not in _SELF_LABELS:
        return character
    name = (entry.get('name') or '').strip()
    return name or None


def _extract_capitalized(text: str) -> list[str]:
    """Extract likely proper nouns from a description by finding capitalized words
    that are NOT sentence starters (those are just grammar, not proper nouns).

    Minimum length of 5 is intentional: short capitalized words like 'May', 'Bay',
    'New', 'Long', 'Old' are common English words that are also sometimes proper
    nouns. Boosting them as Whisper hotwords distorts the decoder's probability
    distribution for normal speech and causes timestamp ordering issues.
    Words of 5+ chars are far more likely to be genuine distinctive proper nouns.
    """
    if not text:
        return []
    starters = set(re.findall(r'(?:^|[.!?])\s*([A-Z][a-zA-Z]+)', text))
    all_caps = re.findall(r'\b([A-Z][a-zA-Z]{4,})\b', text)  # 5+ chars total
    seen: set[str] = set()
    result = []
    for word in all_caps:
        if word in starters:
            continue
        lw = word.lower()
        if lw not in seen:
            seen.add(lw)
            result.append(word)
    return result


def _tmdb_get(path: str, params: dict) -> dict:
    params = dict(params)
    params['api_key'] = tmdb_api_key
    url = f"https://api.themoviedb.org/3{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _build_context_prompt(title_line: str, characters: list[str], keywords: list[str]) -> dict:
    """Build a compact initial_prompt from title + key proper nouns.

    hotwords is intentionally not used: even short lists of rare proper nouns cause
    faster-whisper to significantly distort the token probability distribution on
    every decoding step, which produces non-monotonic timestamp tokens and triggers
    stable-whisper's "timestamps out of order" warning. Common words (e.g. "Self")
    are safe because they are already near peak probability and the boost is trivial.

    initial_prompt is safe at ≤30 tokens because Whisper treats it as previously
    transcribed speech, providing vocabulary context for the first ~30 s segment
    without disturbing the timestamp predictor. Format it as a natural sentence so
    Whisper continues in the same style rather than treating it as an instruction.
    """
    terms = [t for t in characters[:6] + keywords[:3] if t]
    all_parts = [title_line] + terms
    prompt = ', '.join(all_parts) + '.'
    # Stay well under 224 tokens; ~200 chars ≈ 50 tokens
    if len(prompt) > 200:
        prompt = prompt[:200].rsplit(' ', 1)[0]
    return {'prompt': prompt}


def fetch_media_context(parsed: dict | None) -> dict | None:
    """Query TMDB and return {'prompt': str, 'hotwords': str}, or None on failure/no key."""
    if not tmdb_api_key or not parsed:
        return None
    try:
        if parsed['type'] == 'movie':
            return _fetch_movie_context(parsed['title'], parsed['year'])
        else:
            return _fetch_tv_context(parsed['show'], parsed['season'], parsed['episode'], parsed.get('year'))
    except Exception as e:
        logging.warning(f"TMDB lookup failed, skipping prompt enrichment: {e}")
        return None


def _fetch_movie_context(title: str, year: int) -> str | None:
    data = _tmdb_get('/search/movie', {'query': title, 'year': year, 'language': 'en-US'})
    if not data.get('results'):
        data = _tmdb_get('/search/movie', {'query': title, 'language': 'en-US'})
    if not data.get('results'):
        return None

    movie_id = data['results'][0]['id']
    # Fetch credits (for character names) and keywords (topic terms) in one request
    details = _tmdb_get(f'/movie/{movie_id}', {'append_to_response': 'credits,keywords', 'language': 'en-US'})

    characters = [_cast_name(c) for c in details.get('credits', {}).get('cast', [])[:15]]
    characters = [c for c in characters if c]
    keywords = [k['name'] for k in details.get('keywords', {}).get('keywords', [])[:12]]
    keywords += _extract_capitalized(details.get('overview', ''))

    return _build_context_prompt(title, characters, keywords)


def _fetch_tv_context(show: str, season: int, episode: int, year: int | None = None) -> str | None:
    params = {'query': show, 'language': 'en-US'}
    if year:
        params['first_air_date_year'] = year
    data = _tmdb_get('/search/tv', params)
    # If year-filtered search returns nothing, retry without it
    if not data.get('results') and year:
        data = _tmdb_get('/search/tv', {'query': show, 'language': 'en-US'})
    if not data.get('results'):
        return None

    series_id = data['results'][0]['id']
    # Fetch credits and keywords in one request
    series_details = _tmdb_get(f'/tv/{series_id}', {'append_to_response': 'credits,keywords', 'language': 'en-US'})

    characters = [_cast_name(c) for c in series_details.get('credits', {}).get('cast', [])[:15]]
    characters = [c for c in characters if c]
    keywords = [k['name'] for k in series_details.get('keywords', {}).get('results', [])[:12]]
    keywords += _extract_capitalized(series_details.get('overview', ''))

    try:
        ep_data = _tmdb_get(f'/tv/{series_id}/season/{season}/episode/{episode}', {'language': 'en-US'})
        ep_name = ep_data.get('name', '')
        # Also pull guest stars for this specific episode — they're often the focus
        guest_chars = [_cast_name(g) for g in ep_data.get('guest_stars', [])[:5]]
        guest_chars = [c for c in guest_chars if c]
        characters = (guest_chars + characters)[:15]
        # Episode overview often names the specific characters/locations featured
        keywords += _extract_capitalized(ep_data.get('overview', ''))
    except Exception:
        ep_name = ''

    # Split bilingual/slash episode titles (e.g. "Day One/Välkommen") into
    # separate terms so each part contributes vocabulary without punctuation noise.
    ep_parts = [p.strip() for p in ep_name.split('/') if p.strip()] if ep_name else []

    return _build_context_prompt(show.title(), ep_parts + characters, keywords)


app = FastAPI()
model = None

in_docker = os.path.exists('/.dockerenv')
docker_status = "Docker" if in_docker else "Standalone"
last_print_time = None

# deduplicated queue taken from 
class DeduplicatedQueue(queue.Queue):
    """Queue that prevents duplicates in both queued and in-progress tasks."""
    def __init__(self):
        super().__init__()
        self._queued = set()    # Tracks paths in the queue
        self._processing = set()  # Tracks paths being processed
        self._lock = Lock()     # Ensures thread safety

    def put(self, item, block=True, timeout=None):
        with self._lock:
            path = item["path"]
            if path not in self._queued and path not in self._processing:
                super().put(item, block, timeout)
                self._queued.add(path)

    def get(self, block=True, timeout=None):
        item = super().get(block, timeout)
        with self._lock:
            path = item["path"]
            self._queued.discard(path)  # Remove from queued set
            self._processing.add(path)  # Mark as in-progress
        return item

    def task_done(self):
        super().task_done()
        with self._lock:
            # Assumes task_done() is called after processing the item from get()
            # If your workers process multiple items per get(), adjust logic here
            if self.unfinished_tasks == 0:
                self._processing.clear()  # Reset when all tasks are done

    def is_processing(self):
        """Return True if any tasks are being processed."""
        with self._lock:
            return len(self._processing) > 0

    def is_idle(self):
        """Return True if queue is empty AND no tasks are processing."""
        return self.empty() and not self.is_processing()

    def get_queued_tasks(self):
        """Return a list of queued task paths."""
        with self._lock:
            return list(self._queued)

    def get_processing_tasks(self):
        """Return a list of paths being processed."""
        with self._lock:
            return list(self._processing)

#start queue
global task_queue
task_queue = DeduplicatedQueue()

# Ensures only one transcription runs at a time. Without this, parallel
# requests race on the shared model and delete_model() from one request
# can null out the model mid-transcription in another.
_transcription_lock = threading.Lock()

def transcription_worker():
    while True:
        task = task_queue.get()
        logging.debug(f"There are {task_queue.qsize()} tasks left in the queue.")

for _ in range(concurrent_transcriptions):
    threading.Thread(target=transcription_worker, daemon=True).start()

# Define a filter class
class MultiplePatternsFilter(logging.Filter):
    def filter(self, record):
        # Define the patterns to search for
        patterns = [
            "Compression ratio threshold is not met",
            "Processing segment at",
            "Log probability threshold is",
            "Reset prompt",
            "Attempting to release",
            "released on ",
            "Attempting to acquire",
            "acquired on",
            "header parsing failed",
            "timescale not set",
            "misdetection possible",
            "srt was added",
            "doesn't have any audio to transcribe",
        ]
        # Return False if any of the patterns are found, True otherwise
        return not any(pattern in record.getMessage() for pattern in patterns)

# Configure logging
if debug:
    level = logging.DEBUG
    logging.basicConfig(stream=sys.stderr, level=level, format="%(asctime)s %(levelname)s: %(message)s")
else:
    level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(level)  # Set the logger level

for handler in logger.handlers:
    handler.addFilter(MultiplePatternsFilter())

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

#This forces a flush to print progress correctly
def progress(seek, total):
    sys.stdout.flush()
    sys.stderr.flush()
    if(docker_status) == 'Docker':
        global last_print_time
        # Get the current time
        current_time = time.time()
    
        # Check if 5 seconds have passed since the last print
        if last_print_time is None or (current_time - last_print_time) >= 5:
            # Update the last print time
            last_print_time = current_time
            # Log the message
            logging.info("")
            processing = task_queue.get_processing_tasks()[0]
            logging.info(f"Processing file: {processing}")

TIME_OFFSET = 5

@app.get("/status")
def status():
    return {"version" : f"slim-bazarr-subgen {subgen_version}, stable-ts {stable_whisper.__version__}, faster-whisper {faster_whisper.__version__} ({docker_status})"}
   
# idea and some code for asr and detect language from https://github.com/ahmetoner/whisper-asr-webservice
@app.post("//asr")
@app.post("/asr")
def asr(
        task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
        language: Union[str, None] = Query(default=None),
        video_file: Union[str, None] = Query(default=None),
        audio_file: UploadFile = File(...),
):
    task_queued = False
    result = None
    # Read audio before acquiring the lock so the upload doesn't hold up
    # other requests from even starting their upload.
    audio_data = np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0
    _transcription_lock.acquire()
    try:
        logging.info(f"Transcribing '{language}' (Bazarr) from ASR webhook")
        random_name = ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6))

        task_id = { 'path': f"Bazarr-asr-{random_name}" }
        task_queue.put(task_id)
        task_queued = True

        start_model()

        if force_detected_language_to:
            language = force_detected_language_to
            logging.info(f"Language forced to: {language}")
        else:
            # Bazarr passes 'language' from its own per-series database, which can
            # be wrong (e.g. a series configured as Swedish but audio is English).
            # Detect it ourselves and override Bazarr's value; only fall back to
            # what Bazarr sent if detection itself fails.
            detected = _detect_language(audio_data)
            if detected:
                # faster-whisper returns a language code ('ru', 'en', …) directly.
                # Pass it straight to transcribe — no dict lookup needed.
                if language and language != detected:
                    logging.info(
                        f"Language mismatch — Bazarr sent '{language}', "
                        f"detected '{whisper_languages.get(detected, detected)}' ('{detected}'). "
                        f"Using detected language."
                    )
                language = detected
            else:
                logging.info(f"Language detection failed, using Bazarr value: {language}")

        # Build whisper parameters, optionally enriched with TMDB metadata.
        # Only initial_prompt is used — hotwords cause timestamp drift even with
        # short proper-noun lists, as the log-prob boost distorts Whisper's decoder.
        prompt = custom_model_prompt
        if video_file:
            parsed = parse_video_filename(video_file)
            if parsed:
                media_context = fetch_media_context(parsed)
                if media_context:
                    logging.info(f"Enriching whisper context with TMDB data for: {os.path.basename(video_file)}")
                    ctx_prompt = media_context.get('prompt', '')
                    prompt = f"{custom_model_prompt} {ctx_prompt}".strip() if custom_model_prompt else ctx_prompt
                    logging.info(f"Whisper initial_prompt: {prompt}")

        start_time = time.time()
        transcribe_kwargs = dict(
            task=task, input_sr=16000, language=language,
            progress_callback=progress, initial_prompt=prompt,
        )
        if custom_regroup:
            transcribe_kwargs['regroup'] = custom_regroup
        result = _transcribe_with_cpu_fallback(audio_data, transcribe_kwargs)
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        logging.info(f"Bazarr transcription is completed, it took {minutes} minutes and {seconds} seconds to complete.")
    except Exception as e:
        logging.exception(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}")
    finally:
        if task_queued:
            task_queue.task_done()
        delete_model()
        _transcription_lock.release()
    if result:
        return StreamingResponse(
            iter(result.to_srt_vtt(filepath = None, word_level=word_level_highlight)),
            media_type="text/plain",
            headers={
                'Source': 'Transcribed using stable-ts from Subgen!',
            })
    else:
        return

def _detect_language(audio_data: np.ndarray) -> str | None:
    """Run best-of-three language detection at 15 %, 50 % and 80 % of the audio.

    Returns the majority-vote language name (e.g. 'english'), or None on failure.
    Uses faster-whisper's dedicated detect_language() which only needs 30 s of
    audio and is far cheaper than a full transcription.
    """
    total_samples = len(audio_data)
    window = 30 * 16000  # 30 s at 16 kHz — Whisper's native detection window
    votes = []
    for frac in (0.15, 0.50, 0.80):
        start = min(int(total_samples * frac), max(0, total_samples - window))
        chunk = audio_data[start:start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        detect_result = model.detect_language(chunk)
        logging.debug(f"detect_language returned {len(detect_result)} values: {detect_result!r}")
        lang = detect_result[0]
        votes.append(lang)
    winner = Counter(votes).most_common(1)[0][0]
    logging.info(f"Language detection votes: {' / '.join(votes)} → {winner}")
    return winner


@app.post("//detect-language")
@app.post("/detect-language")
def detect_language(
        audio_file: UploadFile = File(...),
):
    global whisper_model
    detected_language = ""
    language_code = ""
    try:
        start_model()
        random_name = ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6))
        task_id = {'path': f"Bazarr-detect-language-{random_name}"}
        task_queue.put(task_id)

        audio_data = np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0
        language_code = _detect_language(audio_data) or ""
        detected_language = whisper_languages.get(language_code, language_code)

    except Exception as e:
        logging.exception(f"Error detecting language for {audio_file.filename}: {e}")

    finally:
        task_queue.task_done()
        delete_model()
        return {"detected_language": detected_language, "language_code": language_code}

def _reload_model_on_cpu():
    """Free the current model from VRAM and reload it on CPU."""
    global model
    try:
        model.model.unload_model()
    except Exception:
        pass
    model = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    logging.info("Loading model on CPU...")
    model = stable_whisper.load_faster_whisper(
        whisper_model, download_root=model_location,
        device='cpu', cpu_threads=whisper_threads,
        num_workers=1, compute_type='default',
    )

def _transcribe_with_cpu_fallback(audio_data, transcribe_kwargs):
    """Run transcription; on CUDA OOM reload the model on CPU and retry once."""
    global model
    if model is None:
        start_model()
    try:
        return model.transcribe(audio_data, **transcribe_kwargs)
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        logging.warning("CUDA out of memory during transcription — falling back to CPU...")
        del e  # clear traceback so gc can free the model tensors
        _reload_model_on_cpu()
        logging.info("Retrying transcription on CPU...")
        return model.transcribe(audio_data, **transcribe_kwargs)

def start_model():
    global model
    if model is None:
        logging.debug("Model was purged, need to re-create")
        try:
            model = stable_whisper.load_faster_whisper(whisper_model, download_root=model_location, device=transcribe_device, cpu_threads=whisper_threads, num_workers=concurrent_transcriptions, compute_type=compute_type)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            logging.warning("CUDA out of memory during model load — falling back to CPU...")
            del e
            _reload_model_on_cpu()

def delete_model():
    if task_queue.qsize() == 0:
        global model
        logging.debug("Queue is empty, clearing/releasing VRAM")
        model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

whisper_languages = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

if __name__ == "__main__":
    import uvicorn
    update_env_variables()
    log_str = f"slim-bazarr-subgen v{subgen_version} | {transcribe_device}"
    if transcribe_device == 'cpu':
        log_str += f" @ {whisper_threads} threads"

    print()
    print(log_str)
    print()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(webhookport), reload=reload_script_on_change, log_level='error', use_colors=True)
