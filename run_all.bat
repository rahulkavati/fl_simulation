@echo off
cd /d C:\Users\rkavati1\Documents\Programming\fl_simulation

for %%C in (10 50 100 200 500) do (
  for %%R in (10 20 30 40 50 60) do (
    start "clients=%%C rounds=%%R" cmd /k ^
      "cd /d C:\Users\rkavati1\Documents\Programming\fl_simulation && python federated_learning_pipeline.py --rounds %%R --clients %%C"
  )
)
