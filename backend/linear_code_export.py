from __future__ import annotations

from typing import Iterable, Optional

from code_export import (
    SUPPORTED_LOSSES,
    SUPPORTED_OPTIMIZERS,
    _filename_token,
    _finite_float,
    _format_float,
    _format_float_list,
    _format_optional_int,
    _loss_code,
    _optimizer_code,
)


def generate_linear_pytorch_script(
    *,
    target_a: float,
    target_b: float,
    initial_a: float,
    initial_b: float,
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
    """Generate a standalone, Colab-ready reproduction of one linear training run."""

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
    initial_a = _finite_float(initial_a, "Initial A")
    initial_b = _finite_float(initial_b, "Initial B")
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
Linear Regression Playground - PyTorch experiment export

This standalone script reproduces the TRAINING run from the website using the
same target coefficients, the exact realized starting A/B values, the same
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

# Target line: y = A*x + B
TARGET_A = {_format_float(target_a)}
TARGET_B = {_format_float(target_b)}

# Exact parameters the website model had BEFORE epoch 1.
# Even if Random initialization was selected, the realized values are written
# here explicitly so rerunning this file starts from the same model.
INITIAL_A = {_format_float(initial_a)}
INITIAL_B = {_format_float(initial_b)}
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
MODEL_PATH = "linear_regression_model.pth"

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
# TARGET_A and TARGET_B are used ONLY to generate the correct y-values.
# They are never passed to the model or optimizer as values to "move toward".
# The model learns its own A/B solely through predictions, loss, gradients,
# and optimizer updates.
y = TARGET_A * X + TARGET_B

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
    title="Linear Regression",
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

class LinearRegressionModel(nn.Module):
    """Learnable model: y = A*x + B."""

    def __init__(self, initial_a, initial_b):
        super().__init__()

        self.A = nn.Parameter(torch.tensor([initial_a], dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor([initial_b], dtype=torch.float32))

    def forward(self, x):
        return self.A * x + self.B


model = LinearRegressionModel(
    initial_a=INITIAL_A,
    initial_b=INITIAL_B,
).to(DEVICE)

print("\\nStarting parameters")
print("-------------------")
print(f"A0 = {{model.A.item():.6f}}")
print(f"B0 = {{model.B.item():.6f}}")


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

    if epoch == 1 or epoch % EVALUATE_EVERY == 0 or epoch == EPOCHS:
        print(f"Epoch {{epoch:,}} / {{EPOCHS:,}}")
        print(f"Train loss: {{displayed_train_loss.item():.8f}}")
        print(f"Test loss:  {{test_loss.item():.8f}}")
        print(f"A = {{model.A.item():.6f}}")
        print(f"B = {{model.B.item():.6f}}")
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

print("\\nLearned coefficients")
print(f"A = {{model.A.item():.6f}}")
print(f"B = {{model.B.item():.6f}}")
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

plt.axhline(TARGET_A, color="tab:blue", linestyle="--", label="Target A")
plt.axhline(TARGET_B, color="tab:orange", linestyle="--", label="Target B")

plt.xlabel("Epoch")
plt.ylabel("Parameter value")
plt.title("How A and B Changed During Training")
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


def build_linear_export_filename(*, optimizer_name: str, loss_function: str, epochs: int) -> str:
    safe_optimizer = _filename_token(optimizer_name)
    safe_loss = _filename_token(loss_function)
    return f"linear_regression_{safe_optimizer}_{safe_loss}_{int(epochs)}epochs.py"
