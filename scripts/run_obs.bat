@echo off
rem Webcam Reactions - OBS mode: streams to an OBS Browser source instead of the virtual camera
rem OBS source URL: http://127.0.0.1:8765/
call "%~dp0run.bat" --obs %*
