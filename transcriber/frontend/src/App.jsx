import React, { useState, useRef, useEffect } from 'react';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcriptionText, setTranscriptionText] = useState('');
  const [language, setLanguage] = useState('ru');
  const [audioSource, setAudioSource] = useState('microphone');
  const [engine, setEngine] = useState('openai');
  const [openaiPrompt, setOpenaiPrompt] = useState('Задача: транскрибировать только речь спикера из входного аудио. Требования: 1) Игнорируй любые вставки, субтитры, заставки и фразы вроде "Субтитры делал DimaTorzok", "Продолжение следует", "[музыка]", "[аплодисменты]". 2) Не добавляй ничего от себя. 3) Сохраняй пунктуацию естественной речи. 4) Сохраняй язык исходной речи. 5) Убирай эээ/ммм/междометия и шумовые слова.');
  const [systemCaptureMode, setSystemCaptureMode] = useState('device'); // 'device' | 'screen'
  const [status, setStatus] = useState('Готов к работе');
  const [error, setError] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [availableDevices, setAvailableDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [selectedSystemDevice, setSelectedSystemDevice] = useState('');
  const [audioLevel, setAudioLevel] = useState(0);
  
  const mediaRecorderRef = useRef(null);
  const sendIntervalRef = useRef(null);
  const segmentedModeRef = useRef(false);
  const websocketRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const levelIntervalRef = useRef(null);

  // Подключение к WebSocket
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      stopAudioLevelMonitoring();
    };
  }, []);

  // Получение списка аудио устройств
  useEffect(() => {
    getAudioDevices();
  }, []);

  const getAudioDevices = async () => {
    try {
      console.log('🔍 Получаем список аудио устройств...');
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(device => device.kind === 'audioinput');
      const audioOutputs = devices.filter(device => device.kind === 'audiooutput');
      
      console.log(`📱 Найдено аудио устройств: ${audioInputs.length}`);
      audioInputs.forEach((device, index) => {
        console.log(`📱 ${index + 1}. ${device.label || `Микрофон ${index + 1}`} (${device.deviceId.substr(0, 8)})`);
      });
      
      setAvailableDevices([...audioInputs, ...audioOutputs]);
      
      // Автоматически выбираем первый микрофон
      if (audioInputs.length > 0) {
        setSelectedDevice(audioInputs[0].deviceId);
        console.log(`✅ Автоматически выбран микрофон: ${audioInputs[0].label || 'Первый микрофон'}`);
      }
      if (audioOutputs.length > 0) {
        setSelectedSystemDevice(audioOutputs[0].deviceId);
        console.log(`✅ Автоматически выбран системный источник: ${audioOutputs[0].label || 'Первый выход'}`);
      }
    } catch (error) {
      console.error('❌ Ошибка получения устройств:', error);
      setError('Не удалось получить список микрофонов');
    }
  };

  const startAudioLevelMonitoring = (stream) => {
    try {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const microphone = audioContextRef.current.createMediaStreamSource(stream);
      
      analyserRef.current.fftSize = 256;
      microphone.connect(analyserRef.current);
      
      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      levelIntervalRef.current = setInterval(() => {
        analyserRef.current.getByteFrequencyData(dataArray);
        
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const average = sum / bufferLength;
        const percentage = Math.round((average / 255) * 100);
        
        setAudioLevel(percentage);
      }, 100);
      
      console.log('✅ Мониторинг уровня звука запущен');
    } catch (error) {
      console.error('❌ Ошибка мониторинга уровня:', error);
    }
  };

  const stopAudioLevelMonitoring = () => {
    if (levelIntervalRef.current) {
      clearInterval(levelIntervalRef.current);
      levelIntervalRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setAudioLevel(0);
  };

  const connectWebSocket = () => {
    try {
      websocketRef.current = new WebSocket('ws://localhost:8001/ws');
      
      websocketRef.current.onopen = () => {
        console.log('🔗 WebSocket подключен');
        setIsConnected(true);
        setStatus('Подключено к серверу');
        setError('');
      };

      websocketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📨 Получено сообщение:', data);
          
          if (data.text) {
            setTranscriptionText(prev => {
              const newText = prev + '\n' + data.text;
              return newText;
            });
            setStatus('Получен текст транскрипции');
          } else if (data.error) {
            setError(data.error);
            setStatus('Ошибка транскрипции');
          } else if (data.status) {
            setStatus(data.status);
          }
        } catch (error) {
          console.error('❌ Ошибка парсинга сообщения:', error);
        }
      };

      websocketRef.current.onclose = () => {
        setIsConnected(false);
        setStatus('Соединение разорвано');
        setTimeout(connectWebSocket, 3000); // Переподключение через 3 секунды
      };

      websocketRef.current.onerror = (error) => {
        setError('Ошибка подключения к серверу');
        setIsConnected(false);
      };
    } catch (err) {
      setError('Не удалось подключиться к серверу');
      setIsConnected(false);
    }
  };

  const startRecording = async () => {
    try {
      console.log('🎤 Начинаем запись...');
      console.log('📡 WebSocket состояние:', websocketRef.current?.readyState);
      setError('');
      setTranscriptionText('');
      
      // Настройка constraints с выбранным устройством
      // Поддержка источника "микрофон + системный звук" (mix)
      let stream;
      if (audioSource === 'both') {
        const mic = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: selectedDevice ? { exact: selectedDevice } : undefined, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
        let sys;
        if (systemCaptureMode === 'screen') {
          try {
            const disp = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
            // Отключим видео, оставим только аудио
            disp.getVideoTracks().forEach(t => t.stop());
            sys = disp;
          } catch (e) {
            console.error('❌ getDisplayMedia не удалось, fallback на устройство', e);
            sys = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: selectedSystemDevice ? { exact: selectedSystemDevice } : undefined } });
          }
        } else {
          sys = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: selectedSystemDevice ? { exact: selectedSystemDevice } : undefined } });
        }
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const dest = ctx.createMediaStreamDestination();
        const micNode = ctx.createMediaStreamSource(mic);
        const sysNode = ctx.createMediaStreamSource(sys);
        micNode.connect(dest);
        sysNode.connect(dest);
        // Преобразуем dest в моно
        const mix = dest.stream;
        stream = mix;
      } else {
        if (audioSource === 'system') {
          if (systemCaptureMode === 'screen') {
            try {
              const disp = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
              disp.getVideoTracks().forEach(t => t.stop());
              stream = disp;
            } catch (e) {
              console.error('❌ getDisplayMedia не удалось для system', e);
              stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: selectedSystemDevice ? { exact: selectedSystemDevice } : undefined } });
            }
          } else {
            stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: selectedSystemDevice ? { exact: selectedSystemDevice } : undefined } });
          }
        } else {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        }
      }

      streamRef.current = stream;
      console.log('✅ Доступ к микрофону получен');

      // Запускаем мониторинг уровня звука
      startAudioLevelMonitoring(stream);

      // Уведомляем сервер о начале записи
      if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({
          type: 'start',
          language: language,
          audioSource: audioSource,
          engine: engine,
          prompt: engine === 'openai' ? openaiPrompt : undefined
        }));
      }
      const options = { audioBitsPerSecond: 128000 };
      const preferredFormats = [
        'audio/ogg;codecs=opus',
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4'
      ];
      for (const fmt of preferredFormats) {
        if (MediaRecorder.isTypeSupported(fmt)) {
          options.mimeType = fmt;
          console.log(`✅ Используем формат: ${fmt}`);
          break;
        }
      }

      segmentedModeRef.current = true;
      const startSegment = () => {
        if (!segmentedModeRef.current) return;
        const mr = new MediaRecorder(stream, options);
        mediaRecorderRef.current = mr;
        console.log('✅ MediaRecorder сегмент создан с mimeType:', mr.mimeType);
        mr.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            const reader = new FileReader();
            reader.onload = () => {
              const base64Data = String(reader.result).split(',')[1] || '';
              if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
                websocketRef.current.send(JSON.stringify({
                  type: 'audio',
                  data: base64Data,
                  format: event.data.type,
                  language: language
                }));
              }
            };
            reader.readAsDataURL(event.data);
          }
        };
        mr.onerror = (event) => {
          console.error('❌ Ошибка MediaRecorder:', event);
          setError('Ошибка записи аудио');
          segmentedModeRef.current = false;
          try { mr.stop(); } catch (_) {}
          stopRecording();
        };
        mr.onstop = () => {
          if (segmentedModeRef.current) {
            setTimeout(startSegment, 0);
          }
        };
        mr.start();
        // Закрываем контейнер и отправляем самодостаточный Blob
        setTimeout(() => {
          if (mr.state === 'recording') {
            try { mr.stop(); } catch (_) {}
          }
        }, 2000);
      };
      startSegment();
      setIsRecording(true);
      setStatus('Запись началась...');

    } catch (err) {
      console.error('❌ Ошибка запуска записи:', err);
      setError('Не удалось получить доступ к микрофону: ' + err.message);
      setStatus('Ошибка доступа к микрофону');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    stopAudioLevelMonitoring();

    // Останавливаем сегментированную запись
    segmentedModeRef.current = false;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      try { mediaRecorderRef.current.stop(); } catch (_) {}
    }

    // Уведомляем сервер об остановке записи
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({
        type: 'stop'
      }));
    }

    setIsRecording(false);
    setStatus('Запись остановлена');
  };

  const copyText = async () => {
    if (transcriptionText.trim()) {
      try {
        await navigator.clipboard.writeText(transcriptionText);
        setStatus('Текст скопирован в буфер обмена');
      } catch (err) {
        setError('Не удалось скопировать текст');
      }
    }
  };

  const clearText = () => {
    setTranscriptionText('');
    setStatus('Текст очищен');
  };

  const getLanguageName = (code) => {
    const languages = {
      'ru': 'Русский',
      'en': 'English',
      'uk': 'Українська'
    };
    return languages[code] || code;
  };

  const getAudioSourceName = (source) => {
    const sources = {
      'microphone': 'Микрофон',
      'system': 'Системный звук'
    };
    return sources[source] || source;
  };

  return (
    <div className="app">
      <div className="header">
        <h1>🎤 Транскрибатор речи</h1>
        <p>Преобразование речи в текст в реальном времени с распознаванием спикеров</p>
      </div>

      <div className="controls">
        <div className="control-group">
          <label>Язык распознавания:</label>
          <select 
            value={language} 
            onChange={(e) => setLanguage(e.target.value)}
            disabled={isRecording}
          >
            <option value="ru">Русский</option>
            <option value="en">English</option>
            <option value="uk">Українська</option>
          </select>
        </div>

        <div className="control-group">
          <label>Источник аудио:</label>
          <select 
            value={audioSource} 
            onChange={(e) => setAudioSource(e.target.value)}
            disabled={isRecording}
          >
            <option value="microphone">Микрофон</option>
            <option value="system">Системный звук</option>
            <option value="both">Микрофон + Системный звук</option>
          </select>
        </div>

        {audioSource === 'both' && (
          <div className="control-group">
            <label>Устройство системного звука (VB-Cable и т.п.):</label>
            <select
              value={selectedSystemDevice}
              onChange={(e) => setSelectedSystemDevice(e.target.value)}
              disabled={isRecording}
            >
              {availableDevices.map((device, index) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Источник ${index + 1}`}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="control-group">
          <label>Движок распознавания:</label>
          <select 
            value={engine}
            onChange={(e) => setEngine(e.target.value)}
            disabled={isRecording}
          >
            <option value="openai">OpenAI Whisper API</option>
            <option value="local">Локальная HF модель</option>
          </select>
        </div>

        {engine === 'openai' && (
          <div className="control-group" style={{minWidth: '320px'}}>
            <label>Промпт для OpenAI:</label>
            <textarea
              value={openaiPrompt}
              onChange={(e) => setOpenaiPrompt(e.target.value)}
              rows="3"
              disabled={isRecording}
            />
          </div>
        )}

        <div className="control-group">
          <label>Микрофон:</label>
          <select 
            value={selectedDevice} 
            onChange={(e) => setSelectedDevice(e.target.value)}
            disabled={isRecording}
          >
            {availableDevices.map((device, index) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Микрофон ${index + 1}`}
              </option>
            ))}
          </select>
        </div>

        <button
          className={`record-button ${isRecording ? 'stop' : 'start'}`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={!isConnected}
        >
          {isRecording ? '⏹ Остановить' : '🎙 Начать запись'}
        </button>
      </div>

      {isRecording && (
        <div className="audio-level">
          <label>Уровень звука:</label>
          <div className="level-meter">
            <div 
              className="level-bar" 
              style={{ width: `${audioLevel}%` }}
            ></div>
            <span className="level-text">{audioLevel}%</span>
          </div>
        </div>
      )}

      <div className="status-bar">
        <p>Статус: {status}</p>
        {error && <p className="error">Ошибка: {error}</p>}
      </div>

      <div className="transcription-area">
        <h2>Текст распознавания:</h2>
        <textarea
          value={transcriptionText}
          onChange={(e) => setTranscriptionText(e.target.value)}
          rows="10"
          cols="50"
          placeholder="Текст распознавания будет отображаться здесь..."
        />
        <button onClick={copyText}>Скопировать текст</button>
        <button onClick={clearText}>Очистить текст</button>
      </div>
    </div>
  );
}

export default App;