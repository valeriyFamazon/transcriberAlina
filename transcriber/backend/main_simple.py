import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from typing import Dict, Any
import ctypes
import wave
import contextlib
try:
    import webrtcvad  # type: ignore
    HAS_WEBRTCVAD = True
except Exception:
    webrtcvad = None  # type: ignore
    HAS_WEBRTCVAD = False
from collections import deque
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf
from transformers import pipeline
import openai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import uvicorn
import dotenv

dotenv.load_dotenv()

# Настройка логирования с эмодзи
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурация OpenAI (ключ из окружения)
openai.api_key = os.getenv("OPENAI_API_KEY", "")
if not openai.api_key:
    logger.warning("⚠️ OPENAI_API_KEY не задан в окружении")

app = FastAPI()

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка FFmpeg
transcriber_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ffmpeg_path = os.path.join(transcriber_dir, "ffmpeg.exe")
ffprobe_path = os.path.join(transcriber_dir, "ffprobe.exe")

logger.info(f"🔍 Ищем FFmpeg в: {ffmpeg_path}")
logger.info(f"🔍 Ищем FFprobe в: {ffprobe_path}")

# Проверяем существование файлов
ffmpeg_exists = os.path.exists(ffmpeg_path)
ffprobe_exists = os.path.exists(ffprobe_path)

if ffmpeg_exists:
    AudioSegment.converter = ffmpeg_path
    logger.info("✅ FFmpeg найден и настроен")
else:
    logger.warning("⚠️ FFmpeg не найден в папке transcriber")

if ffprobe_exists:
    AudioSegment.ffprobe = ffprobe_path
    logger.info("✅ FFprobe найден и настроен")
else:
    logger.warning("⚠️ FFprobe не найден в папке transcriber")

# Дополнительная настройка для pydub
try:
    from pydub.utils import which
    # Проверяем, что pydub может найти ffmpeg
    if not which("ffmpeg"):
        logger.info("🔧 Настраиваем pydub для использования локального ffmpeg")
        import pydub
        pydub.AudioSegment.converter = ffmpeg_path
        pydub.AudioSegment.ffprobe = ffprobe_path
        logger.info("✅ Pydub настроен для использования локального ffmpeg")
except Exception as e:
    logger.warning(f"⚠️ Не удалось настроить pydub: {e}")

# Проверяем работоспособность FFmpeg
def test_ffmpeg():
    """Тестирует работоспособность FFmpeg"""
    try:
        if ffmpeg_exists:
            import subprocess
            result = subprocess.run([ffmpeg_path, "-version"], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ FFmpeg работает корректно")
                return True
            else:
                logger.error(f"❌ FFmpeg не работает: {result.stderr}")
                return False
        else:
            logger.error("❌ FFmpeg не найден")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования FFmpeg: {e}")
        return False

# Тестируем FFmpeg при запуске
ffmpeg_working = test_ffmpeg()

"""
Простой режим: получение base64-аудио чанков по WS, конвертация при необходимости в WAV 16kHz mono,
отправка в OpenAI Whisper и возврат текста. Без диаризации.
"""

# Обработка чанков как самостоятельных файлов (без склейки заголовков/кластеров)
MIN_CHUNK_BYTES = 2 * 1024  # минимальный размер чанка для запуска транскрипции

# Хранилище настроек по соединениям
connection_engine: Dict[int, str] = {}

DEFAULT_OPENAI_PROMPT = (
    "Задача: транскрибировать только речь спикера из входного аудио. "
    "Требования: 1) Игнорируй любые вставки, субтитры, заставки и фразы вроде 'Субтитры делал DimaTorzok', 'Продолжение следует', '[музыка]', '[аплодисменты]'. "
    "2) Не добавляй ничего от себя и не дополняй предложения. 3) Сохраняй пунктуацию естественной речи. "
    "4) Сохраняй язык исходной речи (основной: русский). 5) Убирай эээ/ммм/междометия и шумовые слова."
)

# Локальная HF-пайплайн (ленивая загрузка)
_local_asr = None

def get_local_asr():
    global _local_asr
    if _local_asr is None:
        model_name = os.getenv('HF_LOCAL_ASR_MODEL', 'openai/whisper-small')
        logger.info(f"🧠 Загружаем локальную ASR модель: {model_name}")
        _local_asr = pipeline(
            task='automatic-speech-recognition',
            model=model_name,
            device=-1,
        )
    return _local_asr


def map_language_for_whisper(language_code: str) -> str:
    code = (language_code or '').lower()
    mapping = {
        'ru': 'russian',
        'en': 'english',
        'uk': 'ukrainian',
        'ua': 'ukrainian',
    }
    return mapping.get(code, code or 'russian')

def to_short_path(path: str) -> str:
    """Возвращает 8.3 short path на Windows для ASCII-безопасности (ffmpeg), иначе исходный путь."""
    try:
        if os.name != 'nt':
            return path
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(260)
        result = GetShortPathNameW(path, buffer, 260)
        return buffer.value if result else path
    except Exception:
        return path


def butter_highpass(cutoff_hz: float, sample_rate: int, order: int = 2):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def apply_highpass(signal: np.ndarray, sample_rate: int, cutoff_hz: float = 80.0) -> np.ndarray:
    if signal.size == 0:
        return signal
    b, a = butter_highpass(cutoff_hz, sample_rate)
    return lfilter(b, a, signal)


def read_wave(path: str):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path: str, audio: bytes, sample_rate: int):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame:
    def __init__(self, bytes_data: bytes, timestamp: float, duration: float):
        self.bytes = bytes_data
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / (2 * sample_rate))
    while offset + n <= len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def energy_based_vad(audio: bytes, sample_rate: int, frame_ms: int = 30, threshold_db: float = -45.0) -> bytes:
    if not audio:
        return audio
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len <= 0:
        return audio
    kept = []
    pad_frames = 3
    recent_voice = 0
    for i in range(0, len(samples), frame_len):
        frame = samples[i:i+frame_len]
        if frame.size == 0:
            continue
        rms = np.sqrt(np.mean(np.square(frame))) + 1e-8
        db = 20 * np.log10(rms / 32768.0 + 1e-12)
        is_voiced = db > threshold_db
        if is_voiced:
            recent_voice = pad_frames
        if is_voiced or recent_voice > 0:
            kept.append(frame)
        if recent_voice > 0 and not is_voiced:
            recent_voice -= 1
    if not kept:
        return audio
    out = np.concatenate(kept).astype(np.int16).tobytes()
    return out


def vad_collect(audio: bytes, sample_rate: int, aggressiveness: int = 2) -> bytes:
    if HAS_WEBRTCVAD:
        vad = webrtcvad.Vad(aggressiveness)
        frames = list(frame_generator(30, audio, sample_rate))
        ring_buffer = deque(maxlen=int(300 / 30))
        triggered = False
        voiced: list[bytes] = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    voiced.extend(f.bytes for f, _ in ring_buffer)
                    ring_buffer.clear()
            else:
                voiced.append(frame.bytes)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    ring_buffer.clear()
        return b''.join(voiced) if voiced else b''
    else:
        logger.info("ℹ️ webrtcvad недоступен — используем энерго-VAD")
        return energy_based_vad(audio, sample_rate)


def preprocess_wav(input_wav: str) -> str:
    pcm, sr = read_wave(input_wav)
    # high-pass
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    samples = apply_highpass(samples, sr, 80.0)
    samples = np.clip(samples, -1.0, 1.0)
    hp_bytes = (samples * 32767.0).astype(np.int16).tobytes()
    # VAD
    voiced = vad_collect(hp_bytes, sr, aggressiveness=2)
    if not voiced or len(voiced) < 320:  # <10 ms
        return input_wav
    out_path = input_wav.replace('.wav', '_vad.wav')
    write_wave(out_path, voiced, sr)
    return out_path


def clean_transcript_text(text: str) -> str:
    if not text:
        return text
    deny = [
        'субтитры делал dimatorzok',
        'продолжение следует',
        'подпишитесь',
        'лайк',
        'подписывайтесь',
        'спасибо за просмотр'
    ]
    t = text.strip()
    low = t.lower()
    for phrase in deny:
        low = low.replace(phrase, ' ')
    # уберём квадратные вставки типа [музыка], [аплодисменты]
    import re
    low = re.sub(r"\[.*?\]", " ", low)
    # уберём многоточия-заглушки
    low = re.sub(r"(^|\s)(продолжение\s+следует\.*)$", " ", low)
    cleaned = ' '.join(low.split())
    return cleaned

def guess_extension(audio_bytes: bytes, mime: str) -> str:
    """Грубое определение расширения по сигнатуре и MIME."""
    if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
        return '.wav'
    if audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
        return '.webm'
    if audio_bytes.startswith(b'ftyp'):
        return '.mp4'
    if audio_bytes.startswith(b'OggS'):
        return '.ogg'
    m = (mime or '').lower()
    if 'webm' in m:
        return '.webm'
    if 'ogg' in m:
        return '.ogg'
    if 'mp4' in m or 'm4a' in m:
        return '.mp4'
    if 'wav' in m:
        return '.wav'
    return '.wav'

async def transcribe_audio_chunk(audio_data: bytes, audio_format: str, language: str = "ru", engine: str = "openai", prompt: str = None) -> Dict[str, Any]:
    """Отправляет аудио чанк в OpenAI для транскрипции"""
    temp_file = None
    try:
        logger.info(f"🎙️ === НАЧАЛО ТРАНСКРИПЦИИ ===")
        logger.info(f"🎙️ Начинаем транскрипцию чанка размером {len(audio_data)} байт")

        # Минимальная проверка размера
        if len(audio_data) < 1000:
            logger.warning("⚠️ Чанк слишком мал для транскрипции (< 1KB)")
            return {"error": "Чанк аудио слишком мал для распознавания"}
        
        # Создаем папку temp если не существует
        temp_dir = os.path.join(transcriber_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"📁 Используем папку для временных файлов: {temp_dir}")
        
        # Анализируем формат по первым байтам
        logger.info(f"🔍 Анализ первых байт: {list(audio_data[:10])}")
        
        # Определяем формат по первым байтам
        if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
            logger.info("✅ Обнаружен WAV формат")
            file_extension = '.wav'
        elif audio_data.startswith(b'\x1a\x45\xdf\xa3'):
            logger.info("✅ Обнаружен WebM/MKV формат")
            file_extension = '.webm'
        elif audio_data.startswith(b'ftyp'):
            logger.info("✅ Обнаружен MP4 формат")
            file_extension = '.mp4'
        elif audio_data.startswith(b'OggS'):
            logger.info("✅ Обнаружен OGG формат")
            file_extension = '.ogg'
        else:
            logger.warning(f"⚠️ Неизвестный формат, первые байты: {list(audio_data[:10])}")
            # Пытаемся опереться на MIME из клиента
            fmt_lower = (audio_format or '').lower()
            if 'webm' in fmt_lower:
                logger.info("🎵 Определен как WebM с Opus кодеком")
                file_extension = '.webm'
            elif 'ogg' in fmt_lower:
                file_extension = '.ogg'
            elif 'mp4' in fmt_lower or 'm4a' in fmt_lower:
                file_extension = '.mp4'
            elif 'wav' in fmt_lower:
                file_extension = '.wav'
            else:
                logger.warning(f"⚠️ Пробуем сохранить как WAV для универсальности")
                file_extension = '.wav'
        
        # Создаем временный файл с правильным расширением (ASCII-безопасный короткий путь)
        temp_file = os.path.join(temp_dir, f"audio_{int(time.time() * 1000)}{file_extension}")
        logger.info(f"📁 Создали временный файл для OpenAI: {temp_file}")
        
        # Сохраняем аудио данные в файл
        with open(temp_file, 'wb') as f:
            f.write(audio_data)
        logger.info(f"💾 Сохранили {len(audio_data)} байт в {temp_file}")
        safe_temp_file = to_short_path(temp_file)
        
        try:
            # Конвертация напрямую через FFmpeg CLI → WAV 16kHz mono
            logger.info(f"📁 Загружаем аудио из: {temp_file}")
            output_file = os.path.join(temp_dir, f"openai_audio_{int(time.time() * 1000)}.wav")
            safe_output_file = to_short_path(output_file)
            abs_ffmpeg = os.path.abspath(os.path.join(transcriber_dir, 'ffmpeg.exe'))
            abs_ffmpeg_safe = to_short_path(abs_ffmpeg)
            convert_cmd = [
                abs_ffmpeg_safe,
                '-y',
                '-i', safe_temp_file,
                '-ar', '16000',
                '-ac', '1',
                safe_output_file,
            ]
            logger.info(f"🔧 FFmpeg конвертация → WAV: {' '.join(convert_cmd)}")
            import subprocess
            conv = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=30)
            if conv.returncode != 0 or not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                logger.error(f"❌ FFmpeg конвертация не удалась: code={conv.returncode}")
                if conv.stderr:
                    logger.error(f"FFmpeg stderr: {conv.stderr}")
                raise RuntimeError("FFmpeg conversion failed")
            logger.info(f"✅ Конвертировано в WAV: {output_file}")

            # Предобработка WAV (шумодав/VAD), затем отправляем в ASR движок
            async def transcribe_sync() -> Dict[str, Any]:
                try:
                    # VAD/фильтрация
                    final_wav = preprocess_wav(output_file)
                    if engine == 'openai':
                        with open(final_wav, 'rb') as f:
                            loop = asyncio.get_running_loop()
                            response = await loop.run_in_executor(
                                None,
                                lambda: openai.Audio.transcribe(
                                    model="whisper-1",
                                    file=f,
                                    language=language,
                                    temperature=0,
                                    prompt=prompt or DEFAULT_OPENAI_PROMPT,
                                ),
                            )
                            text = getattr(response, "text", None) or response.get("text", "")
                            text = clean_transcript_text(text)
                            return {"text": text, "language": language}
                    elif engine == 'local':
                        asr = get_local_asr()
                        # Читаем WAV в numpy, чтобы не зависеть от внешнего ffmpeg в PATH
                        audio_array, sr = sf.read(final_wav, dtype='float32', always_2d=False)
                        if isinstance(audio_array, tuple):
                            audio_array = audio_array[0]
                        loop = asyncio.get_running_loop()
                        lang = map_language_for_whisper(language)
                        result = await loop.run_in_executor(
                            None,
                            lambda: asr({"array": audio_array, "sampling_rate": sr}, return_timestamps=False, generate_kwargs={"language": lang})
                        )
                        text = result.get('text', '') if isinstance(result, dict) else str(result)
                        text = clean_transcript_text(text)
                        return {"text": text, "language": language}
                    else:
                        return {"error": f"Неизвестный движок: {engine}"}
                except Exception as e:
                    logger.error(f"❌ Ошибка OpenAI: {e}")
                    return {"error": f"OpenAI ошибка: {e}"}

            return await transcribe_sync()
            
            
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {file_extension.upper()} через pydub: {e}")
            logger.error(f"🔍 Тип ошибки: {type(e).__name__}")
            
            # Падение pydub/конвертера → пробуем ещё раз прямой FFmpeg в WAV
            try:
                output_file = os.path.join(temp_dir, f"openai_audio_{int(time.time() * 1000)}.wav")
                safe_output_file = to_short_path(output_file)
                abs_ffmpeg = os.path.abspath(os.path.join(transcriber_dir, 'ffmpeg.exe'))
                abs_ffmpeg_safe = to_short_path(abs_ffmpeg)
                import subprocess
                conv2 = subprocess.run([
                    abs_ffmpeg_safe, '-y', '-i', safe_temp_file, '-ar', '16000', '-ac', '1', safe_output_file
                ], capture_output=True, text=True, timeout=30)
                if conv2.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    logger.info(f"✅ Повторная конвертация в WAV успешна: {output_file}")

                    async def transcribe_sync2() -> Dict[str, Any]:
                        try:
                            final_wav = preprocess_wav(output_file)
                            if engine == 'openai':
                                with open(final_wav, 'rb') as f:
                                    loop = asyncio.get_running_loop()
                                    response = await loop.run_in_executor(
                                        None,
                                        lambda: openai.Audio.transcribe(
                                            model="whisper-1",
                                            file=f,
                                            language=language,
                                            temperature=0,
                                            prompt=prompt or DEFAULT_OPENAI_PROMPT,
                                        ),
                                    )
                                    text = getattr(response, "text", None) or response.get("text", "")
                                    text = clean_transcript_text(text)
                                    return {"text": text, "language": language}
                            elif engine == 'local':
                                asr = get_local_asr()
                                audio_array, sr = sf.read(final_wav, dtype='float32', always_2d=False)
                                if isinstance(audio_array, tuple):
                                    audio_array = audio_array[0]
                                loop = asyncio.get_running_loop()
                                lang = map_language_for_whisper(language)
                                result = await loop.run_in_executor(
                                    None,
                                    lambda: asr({"array": audio_array, "sampling_rate": sr}, return_timestamps=False, generate_kwargs={"language": lang})
                                )
                                text = result.get('text', '') if isinstance(result, dict) else str(result)
                                text = clean_transcript_text(text)
                                return {"text": text, "language": language}
                            else:
                                return {"error": f"Неизвестный движок: {engine}"}
                        except Exception as e:
                            logger.error(f"❌ Ошибка OpenAI: {e}")
                            return {"error": f"OpenAI ошибка: {e}"}

                    return await transcribe_sync2()
                else:
                    logger.error(f"❌ Повторная FFmpeg конвертация не удалась: code={conv2.returncode}")
                    if conv2.stderr:
                        logger.error(f"FFmpeg stderr: {conv2.stderr}")
                    return {"error": "Не удалось конвертировать аудио в WAV"}
            except Exception as fallback_error:
                logger.error(f"❌ Конвертация через FFmpeg не удалась: {fallback_error}")
                return {"error": f"Ошибка конвертации: {fallback_error}"}
        
    except Exception as e:
        logger.error(f"❌ === ОБЩАЯ ОШИБКА ТРАНСКРИПЦИИ ===")
        logger.error(f"❌ Ошибка транскрипции: {e}")
        logger.error(f"🔍 Тип ошибки: {type(e).__name__}")
        logger.error(f"📍 Место ошибки: transcribe_audio_chunk")
        return {"error": f"Ошибка транскрипции: {e}"}
    finally:
        # Удаляем временные файлы после обработки
        logger.info("🧹 === ОЧИСТКА ВРЕМЕННЫХ ФАЙЛОВ ===")
        
        files_to_clean = []
        if 'temp_file' in locals() and os.path.exists(temp_file):
            files_to_clean.append(temp_file)
        if 'output_file' in locals() and os.path.exists(output_file):
            files_to_clean.append(output_file)
        # Удалим потенциальные временные файлы VAD
        try:
            for name in os.listdir(temp_dir):
                if name.endswith('_vad.wav'):
                    candidate = os.path.join(temp_dir, name)
                    # Безопасно удаляем старые VAD-файлы
                    try:
                        files_to_clean.append(candidate)
                    except Exception:
                        pass
        except Exception:
            pass
        
        for file_path in files_to_clean:
            try:
                os.unlink(file_path)
                logger.info(f"🗑️ Удален временный файл: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Не удалось удалить {file_path}: {cleanup_error}")
        
        if not files_to_clean:
            logger.info("ℹ️ Нет временных файлов для удаления")

# Startup initialization убран - логирование теперь в main()

@app.get("/")
async def root():
    return {"message": "Транскрибатор готов к работе (простой режим)"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔗 Новое WebSocket подключение установлено")
    try:
        while True:
            message = await websocket.receive()
            if 'text' not in message:
                continue
            try:
                data = json.loads(message['text'])
                message_type = data.get('type')
                logger.info(f"📨 Получено текстовое сообщение типа: {message_type}")

                if message_type == 'start':
                    engine = data.get('engine', 'openai')
                    prompt = data.get('prompt')
                    connection_engine[id(websocket)] = json.dumps({"engine": engine, "prompt": prompt})
                    await websocket.send_json({"status": "started", "engine": engine})

                elif message_type == 'stop':
                    await websocket.send_json({"status": "stopped"})

                elif message_type == 'audio':
                    audio_base64 = data.get('data', '')
                    audio_format = data.get('format', 'audio/webm')
                    language = data.get('language', 'ru')
                    engine_info = connection_engine.get(id(websocket))
                    engine = 'openai'
                    prompt = None
                    if engine_info:
                        try:
                            info = json.loads(engine_info)
                            engine = info.get('engine', 'openai')
                            prompt = info.get('prompt')
                        except Exception:
                            pass
                    logger.info(f"🎵 Получены base64 аудио данные: {len(audio_base64)} символов, формат: {audio_format}")
                    try:
                        audio_data = base64.b64decode(audio_base64)
                        logger.info(f"✅ Декодировали {len(audio_data)} байт")
                        if len(audio_data) < MIN_CHUNK_BYTES:
                            logger.info("⏳ Чанк слишком мал — ждём следующий")
                            continue
                        result = await transcribe_audio_chunk(audio_data, audio_format, language, engine=engine, prompt=prompt)
                        logger.info(f"✅ Обработка завершена, результат: {list(result.keys())}")
                        await websocket.send_json(result)
                    except Exception as e:
                        logger.error(f"❌ Ошибка декодирования base64: {e}")
                        await websocket.send_json({"error": f"Ошибка декодирования: {e}"})
                else:
                    logger.warning(f"⚠️ Неизвестный тип сообщения: {message_type}")
            except Exception as e:
                logger.error(f"❌ Ошибка обработки текстового сообщения: {e}")
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket соединение разорвано")
    except Exception as e:
        logger.error(f"❌ Ошибка в WebSocket обработке: {e}")
    finally:
        try:
            connection_engine.pop(id(websocket), None)
        except Exception:
            pass

if __name__ == "__main__":
    # Инициализируем глобальный менеджер транскрипции
    # transcription_manager = TranscriptionManager() # This line is removed
    logger.info("🚀 Транскрибатор запущен в простом режиме (без диаризации)")
    logger.info("🌐 Сервер доступен по адресу: http://localhost:8001")
    logger.info("🔗 WebSocket endpoint: ws://localhost:8001/ws")
    uvicorn.run(app, host="0.0.0.0", port=8001) 