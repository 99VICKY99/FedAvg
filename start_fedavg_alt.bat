@echo off
echo Starting FedAvg Environment...
echo.

REM Initialize conda for Windows Command Prompt
call C:\Users\vicky\miniconda3\condabin\conda.bat activate base
call conda activate fedavg

REM Change to the FedAvg directory  
cd /d "c:\Users\vicky\Desktop\MAJOR-1\FedAvg"

echo.
echo ================================
echo FedAvg Environment Ready!
echo ================================
echo Current directory: %cd%
echo.
echo Available commands:
echo   IID:        python fed_avg.py --data_root "../datasets/" --partition_mode iid
echo   Shard:      python fed_avg.py --data_root "../datasets/" --partition_mode shard
echo   Dirichlet:  python fed_avg.py --data_root "../datasets/" --partition_mode dirichlet --dirichlet_alpha 0.1
echo   With WandB: python fed_avg.py --data_root "../datasets/" --partition_mode dirichlet --dirichlet_alpha 0.1 --wandb --exp_name "dirichlet_test"
echo.
echo Choose an option:
echo [1] Run IID experiment
echo [2] Run Shard-based Non-IID experiment  
echo [3] Run Dirichlet Non-IID experiment (alpha=0.1)
echo [4] Run Dirichlet Non-IID with WandB logging
echo [5] Test implementation
echo [6] Just open command prompt
echo.

choice /c 123456 /m "Enter your choice (1-6): "

if errorlevel 6 goto :open_cmd
if errorlevel 5 goto :test_impl
if errorlevel 4 goto :run_dirichlet_wandb
if errorlevel 3 goto :run_dirichlet
if errorlevel 2 goto :run_shard
if errorlevel 1 goto :run_iid

:run_iid
echo.
echo Running IID experiment (All clients participate)...
python fed_avg.py --data_root "../datasets/" --partition_mode iid --n_clients 10 --n_epochs 50 --frac 1.0
goto :end

:run_shard
echo.
echo Running Shard-based Non-IID experiment (All clients participate)...
python fed_avg.py --data_root "../datasets/" --partition_mode shard --n_clients 10 --n_epochs 50 --frac 1.0
goto :end

:run_dirichlet
echo.
echo Running Dirichlet Non-IID experiment (alpha=0.1, All clients participate)...
python fed_avg.py --data_root "../datasets/" --partition_mode dirichlet --dirichlet_alpha 0.1 --n_clients 10 --n_epochs 50 --frac 1.0
goto :end

:run_dirichlet_wandb
echo.
echo Running Dirichlet Non-IID with WandB logging (All clients participate)...
python fed_avg.py --data_root "../datasets/" --partition_mode dirichlet --dirichlet_alpha 0.1 --wandb --exp_name "dirichlet_test" --n_clients 10 --n_epochs 30 --frac 1.0
goto :end

:test_impl
echo.
echo Testing implementation...
python -c "from data.sampler import FederatedSampler; from data.mnist import MNISTDataset; print('✓ All imports successful'); print('✓ Implementation ready!')"
goto :end

:open_cmd
echo.
echo Opening command prompt...
goto :end

:end
echo.
pause

REM Keep the command prompt open
cmd /k
