subgen_version = '24.07.03'

import os
import xml.etree.ElementTree as ET
import threading
import sys
import time
import queue
import logging
import gc
import random
from typing import Union
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse
import numpy as np

import stable_whisper
from stable_whisper import Segment
import whisper

from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import faster_whisper

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
    
    whisper_model = os.getenv('WHISPER_MODEL', 'medium')
    whisper_threads = max(1, os.cpu_count() - 1)
    concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 1))
    transcribe_device = os.getenv('TRANSCRIBE_DEVICE', 'gpu')
    webhookport = int(os.getenv('WEBHOOKPORT', 9000))
    word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
    debug = convert_to_bool(os.getenv('DEBUG', True))
    model_location = '/models'
    transcribe_or_translate = os.getenv('TRANSCRIBE_OR_TRANSLATE', 'transcribe')
    force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', '').lower()
    compute_type = os.getenv('COMPUTE_TYPE', 'auto')
    reload_script_on_change = convert_to_bool(os.getenv('RELOAD_SCRIPT_ON_CHANGE', False))
    custom_model_prompt = os.getenv('CUSTOM_MODEL_PROMPT', '')
    custom_regroup = os.getenv('CUSTOM_REGROUP', 'cm_sl=84_sl=42++++++1')
    detect_language_length = os.getenv('DETECT_LANGUAGE_LENGTH', 240)
   
    if transcribe_device == "gpu":
        transcribe_device = "cuda"

update_env_variables()

app = FastAPI()
model = None

in_docker = os.path.exists('/.dockerenv')
docker_status = "Docker" if in_docker else "Standalone"
last_print_time = None

#start queue
global task_queue
task_queue = queue.Queue()

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
            logging.info("​")

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
        audio_file: UploadFile = File(...),
):
    try:
        logging.info(f"Transcribing {language} from Bazarr/ASR webhook")
        result = None
        random_name = ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6))

        if force_detected_language_to:
            language = force_detected_language_to

        start_time = time.time()
        start_model()
        
        task_id = { 'path': f"Bazarr-asr-{random_name}" }        
        task_queue.put(task_id)
        
        audio_data = np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0
        if custom_regroup:
            result = model.transcribe_stable(audio_data, task=task, input_sr=16000, language=language, progress_callback=progress, initial_prompt=custom_model_prompt, regroup=custom_regroup)
        else:
            result = model.transcribe_stable(audio_data, task=task, input_sr=16000, language=language, progress_callback=progress, initial_prompt=custom_model_prompt)
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        logging.info(f"Bazarr transcription is completed, it took {minutes} minutes and {seconds} seconds to complete.")
    except Exception as e:
        logging.info(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}")
    finally:
        task_queue.task_done()
        delete_model()
    if result:
        return StreamingResponse(
            iter(result.to_srt_vtt(filepath = None, word_level=word_level_highlight)),
            media_type="text/plain",
            headers={
                'Source': 'Transcribed using stable-ts from Subgen!',
            })
    else:
        return

@app.post("//detect-language")
@app.post("/detect-language")
def detect_language(
        audio_file: UploadFile = File(...),
):  
    global whisper_model
    detected_language = ""  # Initialize with an empty string
    language_code = ""  # Initialize with an empty string
    if int(detect_language_length) != 30:
        logging.info(f"Detect language is set to detect on the first {detect_language_length} seconds of the audio.")
    try:
        start_model()
        random_name = ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", k=6))
        
        task_id = { 'path': f"Bazarr-detect-language-{random_name}" }        
        task_queue.put(task_id)
        
        audio_data = np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0
        detected_language = model.transcribe_stable(whisper.pad_or_trim(audio_data, int(detect_language_length) * 16000), input_sr=16000).language
        # reverse lookup of language -> code, ex: "english" -> "en", "nynorsk" -> "nn", ...
        language_code = get_key_by_value(whisper_languages, detected_language)

    except Exception as e:
        logging.info(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}")
        
    finally:
        task_queue.task_done()
        delete_model()

        return {"detected_language": detected_language, "language_code": language_code}

def start_model():
    global model
    if model is None:
        logging.debug("Model was purged, need to re-create")
        model = stable_whisper.load_faster_whisper(whisper_model, download_root=model_location, device=transcribe_device, cpu_threads=whisper_threads, num_workers=concurrent_transcriptions, compute_type=compute_type)

def delete_model():
    if task_queue.qsize() == 0:
        global model
        logging.debug("Queue is empty, clearing/releasing VRAM")
        model = None
        gc.collect()

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

env_variables = {
    "TRANSCRIBE_DEVICE": {"description": "Can transcribe via gpu (Cuda only) or cpu. Takes option of 'cpu', 'gpu', 'cuda'.", "default": "cpu", "value": ""},
    "WHISPER_MODEL": {"description": "Can be: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1','large-v2', 'large-v3', 'large', 'distil-large-v2', 'distil-medium.en', 'distil-small.en'", "default": "medium", "value": ""},
    "CONCURRENT_TRANSCRIPTIONS": {"description": "Number of files it will transcribe in parallel", "default": "2", "value": ""},
    "WORD_LEVEL_HIGHLIGHT": {"description": "Highlights each word as it's spoken in the subtitle.", "default": False, "value": ""},
    "TRANSCRIBE_OR_TRANSLATE": {"description": "Takes either 'transcribe' or 'translate'. Transcribe will transcribe the audio in the same language as the input. Translate will transcribe and translate into English.", "default": "transcribe", "value": ""},
    "COMPUTE_TYPE": {"description": "Set compute-type using the following information: https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md", "default": "auto", "value": ""},
    "DEBUG": {"description": "Provides some debug data that can be helpful to troubleshoot path mapping and other issues. If set to true, any modifications to the script will auto-reload it (if it isn't actively transcoding). Useful to make small tweaks without re-downloading the whole file.", "default": True, "value": ""},
    "FORCE_DETECTED_LANGUAGE_TO": {"description": "This is to force the model to a language instead of the detected one, takes a 2 letter language code.", "default": "", "value": ""},
    "CUSTOM_MODEL_PROMPT": {"description": "You can override the default prompt (See: [prompt engineering in whisper](https://medium.com/axinc-ai/prompt-engineering-in-whisper-6bb18003562d%29) for great examples).","default": "","value": ""},
    "CUSTOM_REGROUP": {"description": "Attempts to regroup some of the segments to make a cleaner looking subtitle. See #68 for discussion. Set to blank if you want to use Stable-TS default regroups algorithm of cm_sp=,* /，_sg=.5_mg=.3+3_sp=.* /。/?/？","default": "cm_sl=84_sl=42++++++1","value": ""},
    "DETECT_LANGUAGE_LENGTH": {"description": "Detect language on the first x seconds of the audio.","default": 30,"value": ""},
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
