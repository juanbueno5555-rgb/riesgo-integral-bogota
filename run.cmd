@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Sentinel - Mapa de probabilidad de incidentes (Bogota)
echo  Proyecto PTIA - Escuela Colombiana de Ingenieria
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No se encontro Python.
    echo         Instala Python 3.10 o superior desde https://www.python.org/downloads/
    echo         y marca la casilla "Add python.exe to PATH" durante la instalacion.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creando entorno virtual .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Entorno virtual ya existe. OK.
)

if not exist ".venv\.deps-ok" (
    echo [2/3] Instalando dependencias ^(puede tardar unos minutos la primera vez^) ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Fallo la instalacion de dependencias. Revisa tu conexion a internet.
        pause
        exit /b 1
    )
    echo ok > ".venv\.deps-ok"
) else (
    echo [2/3] Dependencias ya instaladas. OK.
)

echo [3/3] Iniciando la aplicacion en http://localhost:8501
echo       ^(deja esta ventana abierta; Ctrl+C para detener^)
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
