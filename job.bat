@echo off
del models.txt

cd src\Process-Data
python setup.py

cd ..\Train-Models
python setup.py

cd ..\..
python main.py > output.txt
