@echo off
echo Starting FedAvg Environment...
echo.

REM Initialize conda
call C:\Users\vicky\miniconda3\Scripts\activate.bat C:\Users\vicky\miniconda3

REM Activate the fedavg environment
call conda activate fedavg

REM Change to the FedAvg directory
cd /d "c:\Users\vicky\Desktop\MAJOR-1\FedAvg"

echo.
echo ================================
echo FedAvg Environment Ready!
echo ================================
echo Current directory: %cd%
echo Conda environment: fedavg
echo.
echo You can now run:
echo   python fed_avg.py --data_root "../datasets/"
echo   python fed_avg.py --data_root "../datasets/" --wandb --exp_name "my_experiment"
echo.

REM Keep the command prompt open
cmd /k
