from __future__ import annotations

import math
from typing import Iterable, Optional


SUPPORTED_LOSSES = {"mse", "l1", "smooth_l1", "huber"}
SUPPORTED_OPTIMIZERS = {"sgd", "adam", "adamw", "rmsprop", "adagrad"}


def _finite_float(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _format_float(value: float) -> str:
    """Stable readable Python float literal."""
    value = _finite_float(value, "Numeric export value")
    if value == 0:
        return "0.0"
    return repr(value)


def _format_optional_int(value: Optional[int]) -> str:
    return "None" if value is None else str(int(value))


def _format_float_list(values: Iterable[float], *, per_line: int = 6) -> str:
    values = [_finite_float(value, "Training x-values") for value in values]
    if not values:
        raise ValueError("At least one training x-value is required for export")

    pieces = [_format_float(value) for value in values]
    lines = []
    for index in range(0, len(pieces), per_line):
        lines.append("    " + ", ".join(pieces[index:index + per_line]) + ",")
    return "[\n" + "\n".join(lines) + "\n]"


def _loss_code(loss_function: str, *, beta: float, delta: float) -> tuple[str, str]:
    if loss_function == "mse":
        return 'LOSS_NAME = "MSE Loss"', "loss_fn = nn.MSELoss()"
    if loss_function == "l1":
        return 'LOSS_NAME = "L1 Loss"', "loss_fn = nn.L1Loss()"
    if loss_function == "smooth_l1":
        beta = _finite_float(beta, "Smooth L1 beta")
        if beta <= 0:
            raise ValueError("Smooth L1 beta must be positive")
        return (
            f'LOSS_NAME = "Smooth L1 Loss"\nSMOOTH_L1_BETA = {_format_float(beta)}',
            "loss_fn = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)",
        )
    if loss_function == "huber":
        delta = _finite_float(delta, "Huber delta")
        if delta <= 0:
            raise ValueError("Huber delta must be positive")
        return (
            f'LOSS_NAME = "Huber Loss"\nHUBER_DELTA = {_format_float(delta)}',
            "loss_fn = nn.HuberLoss(delta=HUBER_DELTA)",
        )
    raise ValueError(f"Unsupported loss function: {loss_function}")


def _optimizer_code(
    optimizer_name: str,
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    beta1: float,
    beta2: float,
    amsgrad: bool,
    rmsprop_alpha: float,
    rmsprop_momentum: float,
    rmsprop_centered: bool,
    adagrad_lr_decay: float,
) -> tuple[str, str]:
    learning_rate = _finite_float(learning_rate, "Learning rate")
    weight_decay = _finite_float(weight_decay, "Weight decay")

    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if weight_decay < 0:
        raise ValueError("Weight decay cannot be negative")

    common_constants = (
        f"LEARNING_RATE = {_format_float(learning_rate)}\n"
        f"WEIGHT_DECAY = {_format_float(weight_decay)}"
    )

    if optimizer_name == "sgd":
        momentum = _finite_float(momentum, "SGD momentum")
        if momentum < 0:
            raise ValueError("SGD momentum cannot be negative")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov requires SGD momentum greater than 0")

        constants = (
            'OPTIMIZER_NAME = "SGD"\n'
            f"{common_constants}\n"
            f"SGD_MOMENTUM = {_format_float(momentum)}\n"
            f"SGD_NESTEROV = {bool(nesterov)}"
        )
        construction = """optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=SGD_MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=SGD_NESTEROV,
)"""
        return constants, construction

    if optimizer_name in {"adam", "adamw"}:
        beta1 = _finite_float(beta1, "Adam beta1")
        beta2 = _finite_float(beta2, "Adam beta2")
        if not (0 < beta1 < 1 and 0 < beta2 < 1):
            raise ValueError("Adam beta values must be between 0 and 1")

        class_name = "Adam" if optimizer_name == "adam" else "AdamW"
        constants = (
            f'OPTIMIZER_NAME = "{class_name}"\n'
            f"{common_constants}\n"
            f"BETA_1 = {_format_float(beta1)}\n"
            f"BETA_2 = {_format_float(beta2)}\n"
            f"AMSGRAD = {bool(amsgrad)}"
        )
        construction = f"""optimizer = torch.optim.{class_name}(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA_1, BETA_2),
    weight_decay=WEIGHT_DECAY,
    amsgrad=AMSGRAD,
)"""
        return constants, construction

    if optimizer_name == "rmsprop":
        alpha = _finite_float(rmsprop_alpha, "RMSprop alpha")
        rms_momentum = _finite_float(rmsprop_momentum, "RMSprop momentum")
        if not (0 < alpha <= 1):
            raise ValueError("RMSprop alpha must be in (0, 1]")
        if rms_momentum < 0:
            raise ValueError("RMSprop momentum cannot be negative")

        constants = (
            'OPTIMIZER_NAME = "RMSprop"\n'
            f"{common_constants}\n"
            f"RMSPROP_ALPHA = {_format_float(alpha)}\n"
            f"RMSPROP_MOMENTUM = {_format_float(rms_momentum)}\n"
            f"RMSPROP_CENTERED = {bool(rmsprop_centered)}"
        )
        construction = """optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=LEARNING_RATE,
    alpha=RMSPROP_ALPHA,
    momentum=RMSPROP_MOMENTUM,
    centered=RMSPROP_CENTERED,
    weight_decay=WEIGHT_DECAY,
)"""
        return constants, construction

    if optimizer_name == "adagrad":
        lr_decay = _finite_float(adagrad_lr_decay, "Adagrad learning-rate decay")
        if lr_decay < 0:
            raise ValueError("Adagrad learning-rate decay cannot be negative")

        constants = (
            'OPTIMIZER_NAME = "Adagrad"\n'
            f"{common_constants}\n"
            f"ADAGRAD_LR_DECAY = {_format_float(lr_decay)}"
        )
        construction = """optimizer = torch.optim.Adagrad(
    model.parameters(),
    lr=LEARNING_RATE,
    lr_decay=ADAGRAD_LR_DECAY,
    weight_decay=WEIGHT_DECAY,
)"""
        return constants, construction

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def generate_pytorch_script(
    *,
    target_a: float,
    target_b: float,
    target_c: float,
    initial_a: float,
    initial_b: float,
    initial_c: float,
    initialization_mode: str,
    seed: Optional[int],
    loss_function: str,
    loss_beta: float,
    loss_delta: float,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    beta1: float,
    beta2: float,
    amsgrad: bool,
    rmsprop_alpha: float,
    rmsprop_momentum: float,
    rmsprop_centered: bool,
    adagrad_lr_decay: float,
    epochs: int,
    train_x: Iterable[float],
) -> str:
    """Generate a standalone, Colab-ready reproduction of one training run."""

    if loss_function not in SUPPORTED_LOSSES:
        raise ValueError(f"Unsupported loss function: {loss_function}")
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    if initialization_mode not in {"manual", "random"}:
        raise ValueError("Initialization mode must be 'manual' or 'random'")
    if int(epochs) < 1:
        raise ValueError("Epochs must be at least 1")

    target_a = _finite_float(target_a, "Target A")
    target_b = _finite_float(target_b, "Target B")
    target_c = _finite_float(target_c, "Target C")
    initial_a = _finite_float(initial_a, "Initial A")
    initial_b = _finite_float(initial_b, "Initial B")
    initial_c = _finite_float(initial_c, "Initial C")
    train_x_values = [_finite_float(value, "Training x-values") for value in train_x]

    if len(train_x_values) < 2:
        raise ValueError("At least two training x-values are required for export")

    loss_constants, loss_construction = _loss_code(
        loss_function,
        beta=loss_beta,
        delta=loss_delta,
    )
    optimizer_constants, optimizer_construction = _optimizer_code(
        optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        beta1=beta1,
        beta2=beta2,
        amsgrad=amsgrad,
        rmsprop_alpha=rmsprop_alpha,
        rmsprop_momentum=rmsprop_momentum,
        rmsprop_centered=rmsprop_centered,
        adagrad_lr_decay=adagrad_lr_decay,
    )

    evaluate_every = max(1, min(100, int(epochs) // 20 or 1))
    train_x_literal = _format_float_list(train_x_values)

    script = f'''\
"""
Quadratic Regression Playground - PyTorch experiment export

This standalone script reproduces the TRAINING run from the website using the
same target coefficients, the exact realized starting A/B/C values, the same
loss function, optimizer settings, and number of epochs.

The website trains on every green training point. To make this exported script
more hands-on, it also creates a small read-only evaluation set immediately
after the website's training x-range. Those evaluation points are NEVER used
by loss.backward() or optimizer.step(), so they do not change the training path.
"""

import torch
from torch import nn
import matplotlib.pyplot as plt


# ============================================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================================

# Target quadratic: y = A*x^2 + B*x + C
TARGET_A = {_format_float(target_a)}
TARGET_B = {_format_float(target_b)}
TARGET_C = {_format_float(target_c)}

# Exact parameters the website model had BEFORE epoch 1.
# Even if Random initialization was selected, the realized values are written
# here explicitly so rerunning this file starts from the same model.
INITIAL_A = {_format_float(initial_a)}
INITIAL_B = {_format_float(initial_b)}
INITIAL_C = {_format_float(initial_c)}
INITIALIZATION_MODE = "{initialization_mode}"
SOURCE_RANDOM_SEED = {_format_optional_int(seed)}

# Exact x-values used as the green training data on the website.
TRAIN_X_VALUES = {train_x_literal}

# The extra evaluation set is only for reporting/visualization in this script.
EVALUATION_PROPORTION = 0.20

EPOCHS = {int(epochs)}
EVALUATE_EVERY = {evaluate_every}

{loss_constants}

{optimizer_constants}

SAVE_MODEL = True
MODEL_PATH = "quadratic_regression_model.pth"

# The website backend trains this playground model on CPU. Keeping CPU here
# makes the reproduction as close to the website run as practical.
DEVICE = torch.device("cpu")


# ============================================================================
# 2. BUILD TRAINING + EVALUATION DATA
# ============================================================================

# Infer the website's x spacing and extend the range for a small test set.
DATA_STEP = TRAIN_X_VALUES[1] - TRAIN_X_VALUES[0]
TEST_COUNT = max(1, round(len(TRAIN_X_VALUES) * EVALUATION_PROPORTION))
TEST_X_VALUES = [
    TRAIN_X_VALUES[-1] + DATA_STEP * (index + 1)
    for index in range(TEST_COUNT)
]

# One complete list, then a transparent split into the website training points
# and the extra evaluation points.
X_VALUES = TRAIN_X_VALUES + TEST_X_VALUES
TRAIN_SPLIT_PROPORTION = len(TRAIN_X_VALUES) / len(X_VALUES)
TRAIN_SPLIT_INDEX = len(TRAIN_X_VALUES)

X = torch.tensor(X_VALUES, dtype=torch.float32).unsqueeze(dim=1)

# IMPORTANT:
# TARGET_A, TARGET_B, TARGET_C are used ONLY to generate the correct y-values.
# They are never passed to the model or optimizer as values to "move toward".
# The model learns its own A/B/C solely through predictions, loss, gradients,
# and optimizer updates.
y = TARGET_A * X**2 + TARGET_B * X + TARGET_C

X_train = X[:TRAIN_SPLIT_INDEX].to(DEVICE)
y_train = y[:TRAIN_SPLIT_INDEX].to(DEVICE)

X_test = X[TRAIN_SPLIT_INDEX:].to(DEVICE)
y_test = y[TRAIN_SPLIT_INDEX:].to(DEVICE)

print(f"Device: {{DEVICE}}")
print(f"Training examples: {{len(X_train)}}")
print(f"Evaluation examples: {{len(X_test)}}")
print(f"Training split proportion in this standalone dataset: {{TRAIN_SPLIT_PROPORTION:.3f}}")


# ============================================================================
# 3. VISUALIZATION HELPER
# ============================================================================

def plot_predictions(
    train_x,
    train_y,
    test_x,
    test_y,
    predictions=None,
    title="Quadratic Regression",
):
    train_x = train_x.detach().cpu()
    train_y = train_y.detach().cpu()
    test_x = test_x.detach().cpu()
    test_y = test_y.detach().cpu()

    if predictions is not None:
        predictions = predictions.detach().cpu()

    plt.figure(figsize=(10, 7))
    plt.scatter(train_x, train_y, s=22, label="Training data")
    plt.scatter(test_x, test_y, s=32, label="Target evaluation y")

    if predictions is not None:
        plt.scatter(test_x, predictions, s=32, label="Model predictions")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()


# ============================================================================
# 4. PYTORCH MODEL
# ============================================================================

class QuadraticRegressionModel(nn.Module):
    """Learnable model: y = A*x^2 + B*x + C."""

    def __init__(self, initial_a, initial_b, initial_c):
        super().__init__()

        self.A = nn.Parameter(torch.tensor([initial_a], dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor([initial_b], dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor([initial_c], dtype=torch.float32))

    def forward(self, x):
        return self.A * x**2 + self.B * x + self.C


model = QuadraticRegressionModel(
    initial_a=INITIAL_A,
    initial_b=INITIAL_B,
    initial_c=INITIAL_C,
).to(DEVICE)

print("\\nStarting parameters")
print("-------------------")
print(f"A0 = {{model.A.item():.6f}}")
print(f"B0 = {{model.B.item():.6f}}")
print(f"C0 = {{model.C.item():.6f}}")


# ============================================================================
# 5. BEFORE-TRAINING FORWARD PASS
# ============================================================================

model.eval()
with torch.inference_mode():
    initial_test_predictions = model(X_test)

plot_predictions(
    X_train,
    y_train,
    X_test,
    y_test,
    predictions=initial_test_predictions,
    title="Before Training",
)


# ============================================================================
# 6. LOSS FUNCTION + OPTIMIZER
# ============================================================================

# Selected loss from the website:
{loss_construction}

# Selected optimizer and exact hyperparameters from the website:
{optimizer_construction}


# ============================================================================
# 7. TRAINING + EVALUATION LOOP
# ============================================================================

train_loss_history = []
test_loss_history = []
a_history = []
b_history = []
c_history = []

print("\\nStarting training...")
print("=" * 68)

for epoch in range(1, EPOCHS + 1):
    # ---- TRAIN ----
    model.train()

    train_predictions = model(X_train)
    train_loss = loss_fn(train_predictions, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # ---- EVALUATE WITHOUT UPDATING PARAMETERS ----
    model.eval()
    with torch.inference_mode():
        # Recompute training loss AFTER the optimizer update so this matches
        # the post-update parameters shown for the current epoch.
        displayed_train_predictions = model(X_train)
        displayed_train_loss = loss_fn(displayed_train_predictions, y_train)

        test_predictions = model(X_test)
        test_loss = loss_fn(test_predictions, y_test)

    train_loss_history.append(displayed_train_loss.item())
    test_loss_history.append(test_loss.item())

    a_history.append(model.A.item())
    b_history.append(model.B.item())
    c_history.append(model.C.item())

    if epoch == 1 or epoch % EVALUATE_EVERY == 0 or epoch == EPOCHS:
        print(f"Epoch {{epoch:,}} / {{EPOCHS:,}}")
        print(f"Train loss: {{displayed_train_loss.item():.8f}}")
        print(f"Test loss:  {{test_loss.item():.8f}}")
        print(f"A = {{model.A.item():.6f}}")
        print(f"B = {{model.B.item():.6f}}")
        print(f"C = {{model.C.item():.6f}}")
        print("-" * 68)


# ============================================================================
# 8. FINAL RESULTS + AFTER-TRAINING PLOT
# ============================================================================

model.eval()
with torch.inference_mode():
    final_test_predictions = model(X_test)
    final_test_loss = loss_fn(final_test_predictions, y_test)

print("\\n" + "=" * 68)
print("TRAINING COMPLETE")
print("=" * 68)

print("\\nTarget coefficients")
print(f"A = {{TARGET_A}}")
print(f"B = {{TARGET_B}}")
print(f"C = {{TARGET_C}}")

print("\\nLearned coefficients")
print(f"A = {{model.A.item():.6f}}")
print(f"B = {{model.B.item():.6f}}")
print(f"C = {{model.C.item():.6f}}")
print(f"\\nFinal evaluation loss ({{LOSS_NAME}}): {{final_test_loss.item():.8f}}")

plot_predictions(
    X_train,
    y_train,
    X_test,
    y_test,
    predictions=final_test_predictions,
    title="After Training",
)


# ============================================================================
# 9. LOSS HISTORY
# ============================================================================

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_loss_history, label="Training loss")
plt.plot(range(1, EPOCHS + 1), test_loss_history, label="Evaluation loss")
plt.xlabel("Epoch")
plt.ylabel(LOSS_NAME)
plt.title("Loss During Training")
plt.legend()
plt.grid(alpha=0.25)
plt.show()


# ============================================================================
# 10. PARAMETER HISTORY
# ============================================================================

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), a_history, label="Learned A")
plt.plot(range(1, EPOCHS + 1), b_history, label="Learned B")
plt.plot(range(1, EPOCHS + 1), c_history, label="Learned C")

plt.axhline(TARGET_A, color="tab:blue", linestyle="--", label="Target A")
plt.axhline(TARGET_B, color="tab:orange", linestyle="--", label="Target B")
plt.axhline(TARGET_C, color="tab:green", linestyle="--", label="Target C")

plt.xlabel("Epoch")
plt.ylabel("Parameter value")
plt.title("How A, B, and C Changed During Training")
plt.legend()
plt.grid(alpha=0.25)
plt.show()


# ============================================================================
# 11. SAVE THE TRAINED MODEL
# ============================================================================

if SAVE_MODEL:
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\\nModel state saved to: {{MODEL_PATH}}")
'''

    return script


def build_export_filename(*, optimizer_name: str, loss_function: str, epochs: int) -> str:
    safe_optimizer = _filename_token(optimizer_name)
    safe_loss = _filename_token(loss_function)
    return f"quadratic_regression_{safe_optimizer}_{safe_loss}_{int(epochs)}epochs.py"


def _filename_token(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "experiment"
