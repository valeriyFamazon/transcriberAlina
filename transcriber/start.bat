@echo off
title Транскрибатор - Запуск
echo ========================================
echo      ТРАНСКРИБАТОР РЕЧИ
echo ========================================
echo.
echo Запуск backend и frontend серверов...
echo.

:: Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    echo Установите Python 3.9 и добавьте его в PATH
    pause
    exit /b 1
)

:: Проверка Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Node.js не найден!
    echo Установите Node.js и добавьте его в PATH
    pause
    exit /b 1
)

:: Запуск backend в отдельном окне
echo Запуск backend сервера...
start "Backend Server" cmd /k "cd /d %~dp0backend && python -m pip install -r requirements.txt && python main_simple.py"

:: Ожидание 5 секунд для запуска backend
timeout /t 5 /nobreak >nul

:: Запуск frontend в отдельном окне
echo Запуск frontend сервера...
start "Frontend Server" cmd /k "cd /d %~dp0frontend && npm install && npm run dev"

echo.
echo ========================================
echo Серверы запускаются...
echo.
echo Backend:  http://localhost:8001
echo Frontend: http://localhost:5173
echo.
echo Подождите несколько секунд для полной загрузки
echo ========================================
echo.

:: Ожидание 10 секунд и открытие браузера
timeout /t 10 /nobreak
start http://localhost:5173

echo Готово! Приложение открыто в браузере.
echo Для остановки серверов закройте окна Backend и Frontend.
pause 