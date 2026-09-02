# Regression Playground

Regression Playground is an interactive web application for exploring how PyTorch learns
linear and quadratic regression models through gradient descent.

Users can configure a target model, choose starting parameters, select a loss function and
optimizer, watch the learned parameters change across epochs, test the trained model on new
x-values, and export the completed experiment as standalone PyTorch code.

## Features

- Linear regression: `y = Ax + B`
- Quadratic regression: `y = Ax² + Bx + C`
- Manual or random parameter initialization with optional reproducible seeds
- L1, MSE, Smooth L1, and Huber losses
- SGD, Adam, AdamW, RMSprop, and Adagrad optimizers
- Optimizer-specific hyperparameter controls
- Epoch-by-epoch training replay
- Training-loss and parameter visualizations
- Testing on manual or randomly generated x-values
- MAE, RMSE, MSE, and maximum absolute error metrics
- Point-by-point target-versus-prediction inspection
- Standalone PyTorch code generation and download
- Dark and light themes with persisted user preference

## Tech stack

- **Backend:** Python, FastAPI, PyTorch, NumPy
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Server:** Uvicorn

## Project structure

```text
regression-playground/
├── backend/
│   ├── main.py
│   ├── quadratic_backend.py
│   ├── linear_backend.py
│   ├── code_export.py
│   └── linear_code_export.py
├── frontend/
│   ├── index.html
│   ├── linear.html
│   ├── app.js
│   ├── linear_app.js
│   ├── theme.js
│   └── style.css
├── .gitignore
├── .python-version
├── requirements.txt
├── run.bat
└── README.md
```

## Run locally on Windows

The easiest option is to double-click:

```text
run.bat
```

The launcher creates `.venv` if needed, installs the packages from `requirements.txt`
when required, starts FastAPI/Uvicorn, and opens the site at:

```text
http://127.0.0.1:8000
```

Keep the launcher window open while using the app. Press `Ctrl+C` in that window to stop
the local server.

## Run locally from a terminal

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the app from the backend directory:

```bash
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in a browser.

## How the playground works

The target coefficients are used to generate synthetic target values. They are not handed
directly to the optimizer as the coefficients it should move toward. PyTorch learns model
parameters by minimizing the selected loss through gradient-based optimization.

The Test page evaluates a snapshot of the learned model on new x-values without updating its
parameters.

The Export page generates standalone PyTorch code that records the completed experiment,
including the realized starting parameters, selected loss, optimizer settings, epoch count,
and training inputs.

## Repository notes

Virtual environments, Python cache files, local environment-variable files, generated model
artifacts, logs, and ZIP backups are excluded through `.gitignore`.

No open-source license has been selected yet. Add a license only after deciding how you want
other people to be allowed to reuse or modify the source code.
