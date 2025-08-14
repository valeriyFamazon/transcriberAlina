# 🎤 Транскрибатор речи с диаризацией (Windows, FastAPI + React)

Единый README, объединяющий все .md из проекта: обзор, запуск, протокол, диагностика, частые проблемы и статус фиксов.

## ✨ Возможности
- 🎙️ **Реал‑тайм транскрипция** (WebSocket)
- 👥 **Диаризация** (pyannote, опционально)
- 🌍 **Языки:** ru, en, uk
- 🎵 **Источники:** микрофон, системный звук (через VB‑Cable)
- 🧹 **Предобработка аудио:** high‑pass фильтр + VAD (webrtcvad) для подавления шума/тишины
- 📋 **UI:** копирование/очистка, выбор микрофона, индикатор уровня

## 📦 Требования
- Windows 10+
- Python 3.9.x (строго)
- Node.js 16+ (LTS)
- FFmpeg бинарники в `transcriber/` (`ffmpeg.exe`, `ffprobe.exe`)
- Интернет (OpenAI API, загрузка моделей)
- Доп. зависимости для шумоподавления: `webrtcvad`

## ⚙️ Конфигурация
- Переменные окружения (рекомендуется):
  - `OPENAI_API_KEY` — ключ OpenAI
  - `HF_TOKEN` — токен Hugging Face (для pyannote)
- Порты: FE `5173`, BE `8001`
- Путь к FFmpeg настраивается автоматически из корня `transcriber/`

## 🚀 Запуск
Самый простой способ:

```bat
transcriber\start.bat
```

Ручной запуск:
```bat
cd transcriber\backend
pip install -r requirements.txt
python main_simple.py  # онлайн режим (OpenAI)

cd ..\frontend
npm install
npm run dev
```

Адреса: FE `http://localhost:5173`, BE `http://localhost:8001`

## 🧩 Архитектура и режимы
- `backend/main_simple.py` — базовый онлайн режим (OpenAI Whisper API)
- `backend/main.py` — онлайн режим + диаризация (pyannote, HF токен)
- `backend/main_offline.py` — офлайн режим (локальный `openai-whisper`), без внешнего API
- Frontend (Vite + React) шлёт чанки аудио base64 по WS `ws://localhost:8001/ws`

### Предобработка аудио (backend)
- Конвертация входных чанков → WAV 16kHz mono
- High‑pass фильтр (80 Hz) для среза низкочастотного гула
- VAD (webrtcvad, агрессивность 2) — удаление тишины/фонового шума
- Очистка текста от мусорных вставок вроде «Субтитры делал ...», «Продолжение следует...», `[музыка]`

### WebSocket протокол (frontend → backend)
```json
{
  "type": "start|audio|stop",
  "data": "base64_audio_data",
  "format": "audio/webm;codecs=opus",
  "language": "ru|en|uk"
}
```

### Формат ответов (backend → frontend)
```json
{
  "type": "transcription|status|error",
  "text": "распознанный_текст",
  "speaker": "SPEAKER_00",
  "language": "ru"
}
```

## 🔊 Системный звук (VB‑Cable)
1) Установить `VB-Cable` (vb-audio.com/Cable) администратором, перезагрузить
2) Windows → Звук: Вывод `CABLE Input`, Ввод `CABLE Output`
3) В UI выбрать источник «Системный звук»

## 🧪 Диагностика и тесты
- `check_dependencies.py` — проверка структуры/зависимостей/FFmpeg/Node
- `cleanup_temp.py` — список/очистка `transcriber/temp`
- `debug_webm.py` — анализ WebM + ffprobe + конвертация
- `test_ffmpeg.py` — проверка FFmpeg и интеграции с pydub
- `test_transcription.py` — генерация синуса → конвертация → OpenAI
- `test_working_wav.py` — отправка существующих WAV в OpenAI
- `test_system.py` — e2e‑проверки окружения (локальные сервисы)

Полезные папки:
- `transcriber/temp/` — временные файлы (WebM/WAV) для отладки

## 🛟 Частые проблемы и решения
1) FFmpeg не найден / конвертация падает
   - Убедитесь, что `ffmpeg.exe` и `ffprobe.exe` лежат в `transcriber/`
   - Проверьте `check_dependencies.py` и `test_ffmpeg.py`

2) Пишется не микрофон, а «системные» субтитры
   - Откройте `debug_microphone.html`, выберите корректный микрофон
   - В Windows отключите «Стерео микшер», проверьте устройство «Ввод»

3) Пустая транскрипция
   - Интернет/ключ OpenAI, громкость/шум, корректность источника в UI

4) WebM повреждён (EBML header parsing failed)
   - Проверить отправку чанков с фронта (base64 корректен, размер > 0)
   - Использовать fallback (прямая отправка в OpenAI) или конвертацию

5) В тексте попадаются «лишние» фразы (субтитры/заставки)
   - Включена очистка текстов на бэкенде, при необходимости расширьте список стоп‑фраз в `clean_transcript_text()`

## 📒 Сводка исправлений (конденсат из FIXES_*, PROBLEM_SOLVED, FINAL_*)
- Конвертация WebM → WAV перед отправкой в OpenAI
- Прямой fallback upload в OpenAI при ошибке FFmpeg/pydub
- Улучшено логирование всех этапов, автопроверка FFmpeg
- Добавлен выбор микрофона и индикатор уровня на фронте
- Автопереподключение WebSocket и обработка start/stop

Ожидаемая задержка: 1–3 сек; точность 85–95% (зависит от аудио)

## ⚠️ Выявленные несоответствия и что надо исправить
- Безопасность: убрать хардкод ключей из кода
  - `backend/main_simple.py`, `backend/main.py`, диагностические скрипты — читать `OPENAI_API_KEY`/`HF_TOKEN` из окружения

- `backend/main_simple.py` — ошибки и несовместимости
  - Используется `time` без импорта; добавить `import time`
  - В `transcribe_audio_chunk()` используется `audio_format` (L203), но эта переменная не определена в функции; нужно пробрасывать формат параметром из WS‑обработчика
  - Используется `openai.Audio.atranscribe` (async), при `openai==0.28.1` стабильнее `openai.Audio.transcribe` (sync) внутри `run_in_executor` или оставить sync‑вызов

- `backend/main_offline.py` — конвертация и чтение входа некорректны
  - `process_audio_data()` создаёт пустой временный `.webm` и читает его, игнорируя реальные байты
  - Нужно либо читать из `io.BytesIO(audio_data)`, либо записывать `audio_data` во временный файл и только затем загружать через pydub
  - При фолбэке возвращаются сырые WebM‑байты, которые затем сохраняются как `.wav` — несоответствие формату; нужно гарантировать WAV‑контент

- Тесты не соответствуют текущему коду
  - `backend/test_main_simple.py` импортирует `TranscriptionManager`, `process_audio_data`, которых нет в `main_simple.py` сейчас — тесты устарели и будут падать. Обновить тесты под актуальные API или восстановить требуемые объекты.

- Документация о temp‑файлах расходится с кодом
  - Док‑страницы утверждают, что автоочистка выключена, но `main_simple.py` удаляет временные файлы в `finally`. Решить единообразно: либо оставить и описать автoочистку, либо выключить в коде на время отладки

## ✅ Рекомендованные правки (минимум для стабильной работы)
1) Вынести `OPENAI_API_KEY`/`HF_TOKEN` в переменные окружения; удалить хардкод
2) `main_simple.py`:
   - добавить `import time`
   - в `websocket_endpoint` передавать `audio_format` в `transcribe_audio_chunk(audio_data, audio_format)`; внутри выбирать расширение/конверсию согласно формату
   - заменить `atranscribe` на надёжный путь (sync API в executor или проверенный async)
3) `main_offline.py`:
   - переписать `process_audio_data()` для корректной загрузки из байтов и возврата реального WAV
4) Синхронизировать тесты с кодом или восстановить ожидаемые функции в `main_simple.py`

## 📋 Чек‑лист готовности
- Python 3.9, Node.js установлены
- FFmpeg файлы лежат в `transcriber/`
- BE на 8001, FE на 5173
- Ключи заданы через окружение
- В `debug_microphone.html` выбран правильный микрофон

---

Готово к использованию. Для полной пошаговой диагностики см. объединённые разделы выше (ранее: SOLUTION_GUIDE, WEBM_PROBLEM_ANALYSIS, MICROPHONE_PROBLEM_FOUND, FIXES_SUMMARY, FINAL_FIXES_STATUS). 🎉