@echo off
title Тест системы
echo ========================================
echo      ПРОВЕРКА СИСТЕМЫ
echo ========================================
echo.

echo Проверка Python...
python --version
if errorlevel 1 (
    echo ❌ Python НЕ установлен!
    echo Скачайте Python 3.9 с https://www.python.org/downloads/
) else (
    echo ✅ Python найден
)
echo.

echo Проверка Node.js...
node --version
if errorlevel 1 (
    echo ❌ Node.js НЕ установлен!
    echo Скачайте Node.js с https://nodejs.org/
) else (
    echo ✅ Node.js найден
)
echo.

echo Проверка npm...
npm --version
if errorlevel 1 (
    echo ❌ npm НЕ найден!
) else (
    echo ✅ npm найден
)
echo.

echo Проверка pip...
pip --version
if errorlevel 1 (
    echo ❌ pip НЕ найден!
) else (
    echo ✅ pip найден
)
echo.

echo ========================================
echo Если все компоненты найдены, можете
echo запускать start.bat для запуска приложения
echo ========================================
pause 