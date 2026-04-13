# Detect Operating System
ifeq ($(OS),Windows_NT)
	# Windows variables
	PYTHON_GLOBAL = python
	PYTHON = .venv\Scripts\python.exe
	PIP = .venv\Scripts\pip.exe
	RM_VENV = if exist .venv rmdir /s /q .venv
	RM_EGG = if exist RL_FT.egg-info rmdir /s /q RL_FT.egg-info
	RM_CACHE = for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	RM_LOGS = if exist logs rmdir /s /q logs
	RM_OUTPUTS = if exist outputs rmdir /s /q outputs
	RM_MUJOCO_LOG = if exist MUJOCO_LOG.TXT del /q MUJOCO_LOG.TXT
else
	# Unix/Linux/macOS variables
	PYTHON_GLOBAL = python3
	PYTHON = .venv/bin/python
	PIP = .venv/bin/pip
	RM_VENV = rm -rf .venv
	RM_EGG = rm -rf *.egg-info
	RM_CACHE = find . -type d -name "__pycache__" -exec rm -rf {} +
	RM_LOGS = rm -rf logs
	RM_OUTPUTS = rm -rf outputs
	RM_MUJOCO_LOG = rm -f MUJOCO_LOG.TXT
endif

.PHONY: install clean test

# Creates the virtual environment and installs package in editable mode
install:
	$(PYTHON_GLOBAL) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e .

# Wipes the environment and all cache files
clean:
	$(RM_VENV)
	$(RM_EGG)
	$(RM_CACHE)
	$(RM_LOGS)
	$(RM_OUTPUTS)
	$(RM_MUJOCO_LOG)

test:
	$(PYTHON) -c "import mujoco; import gymnasium; print('Success: MuJoCo and Gymnasium are loaded and ready.')"