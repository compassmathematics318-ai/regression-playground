import math
from typing import Optional

import torch
from torch import nn

from quadratic_backend import _build_loss, _build_optimizer, _to_float_list, _validate_finite


class LinearRegressionModel(nn.Module):
    """Model for y = A*x + B."""

    def __init__(self, initial_a=None, initial_b=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.tensor([initial_a], dtype=torch.float32)
            if initial_a is not None
            else torch.randn(1, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.tensor([initial_b], dtype=torch.float32)
            if initial_b is not None
            else torch.randn(1, dtype=torch.float32)
        )

    def forward(self, x):
        return self.weight * x + self.bias


def train_linear(
    true_a=3.0,
    true_b=1.0,
    initial_a=None,
    initial_b=None,
    seed: Optional[int] = None,
    loss_function="l1",
    loss_beta=1.0,
    loss_delta=1.0,
    optimizer_name="sgd",
    learning_rate=0.01,
    weight_decay=0.0,
    momentum=0.0,
    nesterov=False,
    beta1=0.9,
    beta2=0.999,
    amsgrad=False,
    rmsprop_alpha=0.99,
    rmsprop_momentum=0.0,
    rmsprop_centered=False,
    adagrad_lr_decay=0.0,
    epochs=200,
    start=0.0,
    end=1.0,
    step=0.02,
):
    """
    Train on one synthetic linear dataset.

    Every generated point is a TRAINING point. Testing is intentionally kept
    separate and happens later through ``test_linear`` so the website can
    demonstrate the distinction between fitting and evaluating a model.
    """

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if step <= 0:
        raise ValueError("step must be positive")
    if end <= start:
        raise ValueError("end must be greater than start")

    _validate_finite(
        [
            true_a,
            true_b,
            learning_rate,
            weight_decay,
            loss_beta,
            loss_delta,
            momentum,
            beta1,
            beta2,
            rmsprop_alpha,
            rmsprop_momentum,
            adagrad_lr_decay,
            start,
            end,
            step,
        ],
        "Training inputs",
    )

    device = torch.device("cpu")

    # A supplied seed makes random initialization reproducible. If seed is
    # omitted, PyTorch's RNG state advances naturally for a fresh random start.
    if seed is not None:
        torch.manual_seed(int(seed))

    x_train = torch.arange(
        start, end, step, dtype=torch.float32, device=device
    ).unsqueeze(1)
    if len(x_train) < 2:
        raise ValueError("The selected x-range produces too few training points")

    y_train = true_a * x_train + true_b

    initialization_mode = (
        "random"
        if initial_a is None and initial_b is None
        else "manual"
    )

    model = LinearRegressionModel(
        initial_a=initial_a,
        initial_b=initial_b,
    ).to(device)

    loss_fn, loss_config = _build_loss(
        loss_function,
        beta=loss_beta,
        delta=loss_delta,
    )
    optimizer, optimizer_config = _build_optimizer(
        optimizer_name,
        model.parameters(),
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

    model.eval()
    with torch.inference_mode():
        initial_predictions = model(x_train)

    initial_parameters = {
        "a": model.weight.item(),
        "b": model.bias.item(),
    }

    history = []

    for epoch in range(epochs):
        model.train()

        predictions = model(x_train)
        train_loss = loss_fn(predictions, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            full_predictions = model(x_train)
            displayed_loss = loss_fn(full_predictions, y_train)

        if not torch.isfinite(displayed_loss) or not torch.isfinite(full_predictions).all():
            raise ValueError(
                f"Training diverged at epoch {epoch + 1}. "
                "Try a smaller learning rate or less aggressive optimizer settings."
            )

        history.append(
            {
                "epoch": epoch + 1,
                "a": model.weight.item(),
                "b": model.bias.item(),
                "train_loss": displayed_loss.item(),
                "predictions": _to_float_list(full_predictions),
            }
        )

    final_parameters = {
        "a": model.weight.item(),
        "b": model.bias.item(),
    }

    return {
        "target_parameters": {
            "a": float(true_a),
            "b": float(true_b),
        },
        "initial_parameters": initial_parameters,
        "final_parameters": final_parameters,
        "training_configuration": {
            "initialization": {
                "mode": initialization_mode,
                "seed": int(seed) if seed is not None else None,
            },
            "loss": loss_config,
            "optimizer": optimizer_config,
            "epochs": int(epochs),
            "seed": int(seed) if seed is not None else None,
        },
        "dataset": {
            "x": _to_float_list(x_train),
            "y": _to_float_list(y_train),
            "train_x": _to_float_list(x_train),
            "train_y": _to_float_list(y_train),
            "train_size": len(x_train),
        },
        "initial_predictions": _to_float_list(initial_predictions),
        "history": history,
    }


def test_linear(
    *,
    true_a,
    true_b,
    model_a,
    model_b,
    model_epoch=None,
    x_values=None,
):
    """
    Evaluate a trained linear model on NEW x-values.

    The target y-values come from the theoretically correct line while the
    predicted y-values come from the learned A and B. No parameters are updated
    here: this is evaluation only.
    """

    x_values = [float(value) for value in x_values]
    if not x_values:
        raise ValueError("Provide at least one x-value to test")
    if len(x_values) > 1000:
        raise ValueError("At most 1000 test x-values are allowed at once")

    _validate_finite(
        [true_a, true_b, model_a, model_b, *x_values],
        "Test inputs",
    )

    x_values = sorted(x_values)
    x = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1)

    true_y = true_a * x + true_b
    predicted_y = model_a * x + model_b
    errors = predicted_y - true_y

    abs_errors = torch.abs(errors)
    squared_errors = errors ** 2

    mae = torch.mean(abs_errors).item()
    mse = torch.mean(squared_errors).item()
    rmse = math.sqrt(mse)
    max_abs_error = torch.max(abs_errors).item()

    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        pad = max(1.0, abs(x_min) * 0.25)
    else:
        pad = (x_max - x_min) * 0.08

    curve_min = x_min - pad
    curve_max = x_max + pad
    curve_x = torch.linspace(curve_min, curve_max, 240, dtype=torch.float32).unsqueeze(1)
    target_curve_y = true_a * curve_x + true_b
    prediction_curve_y = model_a * curve_x + model_b

    true_list = _to_float_list(true_y)
    pred_list = _to_float_list(predicted_y)
    error_list = _to_float_list(errors)

    rows = [
        {
            "x": x_values[i],
            "true_y": true_list[i],
            "predicted_y": pred_list[i],
            "error": error_list[i],
            "absolute_error": abs(error_list[i]),
        }
        for i in range(len(x_values))
    ]

    return {
        "target_parameters": {
            "a": float(true_a),
            "b": float(true_b),
        },
        "model_parameters": {
            "a": float(model_a),
            "b": float(model_b),
        },
        "model_epoch": int(model_epoch) if model_epoch is not None else None,
        "metrics": {
            "count": len(x_values),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "max_abs_error": max_abs_error,
        },
        "test_points": {
            "x": x_values,
            "true_y": true_list,
            "predicted_y": pred_list,
            "error": error_list,
        },
        "curve": {
            "x": _to_float_list(curve_x),
            "true_y": _to_float_list(target_curve_y),
            "predicted_y": _to_float_list(prediction_curve_y),
        },
        "rows": rows,
    }


if __name__ == "__main__":
    result = train_linear(epochs=500)
    print("Target:", result["target_parameters"])
    print("Final:", result["final_parameters"])
    print("Final train loss:", result["history"][-1]["train_loss"])
