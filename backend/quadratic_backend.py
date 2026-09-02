import math
from typing import Iterable, Optional

import torch
from torch import nn


class QuadraticRegressionModel(nn.Module):
    """Model for y = A*x^2 + B*x + C."""

    def __init__(self, initial_a=None, initial_b=None, initial_c=None):
        super().__init__()

        self.weight1 = nn.Parameter(
            torch.tensor([initial_a], dtype=torch.float32)
            if initial_a is not None
            else torch.randn(1, dtype=torch.float32)
        )
        self.weight2 = nn.Parameter(
            torch.tensor([initial_b], dtype=torch.float32)
            if initial_b is not None
            else torch.randn(1, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.tensor([initial_c], dtype=torch.float32)
            if initial_c is not None
            else torch.randn(1, dtype=torch.float32)
        )

    def forward(self, x):
        return self.weight1 * (x ** 2) + self.weight2 * x + self.bias


def _to_float_list(tensor):
    """Convert a tensor into a JSON-friendly Python list."""
    return tensor.detach().cpu().flatten().tolist()


def _validate_finite(values: Iterable[float], label: str):
    for value in values:
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must contain only finite numbers")


def _build_loss(loss_function: str, *, beta: float, delta: float):
    if loss_function == "mse":
        return nn.MSELoss(), {
            "key": "mse",
            "label": "MSE Loss",
            "parameters": {},
        }

    if loss_function == "l1":
        return nn.L1Loss(), {
            "key": "l1",
            "label": "L1 Loss",
            "parameters": {},
        }

    if loss_function == "smooth_l1":
        if beta <= 0:
            raise ValueError("Smooth L1 beta must be positive")
        return nn.SmoothL1Loss(beta=beta), {
            "key": "smooth_l1",
            "label": "Smooth L1 Loss",
            "parameters": {"beta": float(beta)},
        }

    if loss_function == "huber":
        if delta <= 0:
            raise ValueError("Huber delta must be positive")
        return nn.HuberLoss(delta=delta), {
            "key": "huber",
            "label": "Huber Loss",
            "parameters": {"delta": float(delta)},
        }

    raise ValueError(f"Unsupported loss function: {loss_function}")


def _build_optimizer(
    optimizer_name: str,
    parameters,
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
):
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if weight_decay < 0:
        raise ValueError("Weight decay cannot be negative")

    common_config = {
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
    }

    if optimizer_name == "sgd":
        if momentum < 0:
            raise ValueError("SGD momentum cannot be negative")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires SGD momentum greater than 0")

        optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        return optimizer, {
            "key": "sgd",
            "label": "SGD",
            "parameters": {
                **common_config,
                "momentum": float(momentum),
                "nesterov": bool(nesterov),
            },
        }

    if optimizer_name in {"adam", "adamw"}:
        if not (0 < beta1 < 1 and 0 < beta2 < 1):
            raise ValueError("Adam beta values must be between 0 and 1")

        optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
        optimizer = optimizer_class(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        return optimizer, {
            "key": optimizer_name,
            "label": "Adam" if optimizer_name == "adam" else "AdamW",
            "parameters": {
                **common_config,
                "beta1": float(beta1),
                "beta2": float(beta2),
                "amsgrad": bool(amsgrad),
            },
        }

    if optimizer_name == "rmsprop":
        if not (0 < rmsprop_alpha <= 1):
            raise ValueError("RMSprop alpha must be in (0, 1]")
        if rmsprop_momentum < 0:
            raise ValueError("RMSprop momentum cannot be negative")

        optimizer = torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
            alpha=rmsprop_alpha,
            momentum=rmsprop_momentum,
            centered=rmsprop_centered,
            weight_decay=weight_decay,
        )
        return optimizer, {
            "key": "rmsprop",
            "label": "RMSprop",
            "parameters": {
                **common_config,
                "alpha": float(rmsprop_alpha),
                "momentum": float(rmsprop_momentum),
                "centered": bool(rmsprop_centered),
            },
        }

    if optimizer_name == "adagrad":
        if adagrad_lr_decay < 0:
            raise ValueError("Adagrad learning-rate decay cannot be negative")

        optimizer = torch.optim.Adagrad(
            parameters,
            lr=learning_rate,
            lr_decay=adagrad_lr_decay,
            weight_decay=weight_decay,
        )
        return optimizer, {
            "key": "adagrad",
            "label": "Adagrad",
            "parameters": {
                **common_config,
                "lr_decay": float(adagrad_lr_decay),
            },
        }

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_quadratic(
    true_a=3.0,
    true_b=1.0,
    true_c=-1.0,
    initial_a=None,
    initial_b=None,
    initial_c=None,
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
    Train on one synthetic quadratic dataset.

    Every generated point is a TRAINING point. Testing is intentionally kept
    separate and happens later through ``test_quadratic`` so the website can
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
            true_c,
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
    # omitted, PyTorch's RNG state advances naturally, producing a fresh random
    # initialization on each request.
    if seed is not None:
        torch.manual_seed(int(seed))

    x_train = torch.arange(
        start, end, step, dtype=torch.float32, device=device
    ).unsqueeze(1)
    if len(x_train) < 2:
        raise ValueError("The selected x-range produces too few training points")

    y_train = true_a * (x_train ** 2) + true_b * x_train + true_c

    initialization_mode = (
        "random"
        if initial_a is None and initial_b is None and initial_c is None
        else "manual"
    )

    model = QuadraticRegressionModel(
        initial_a=initial_a,
        initial_b=initial_b,
        initial_c=initial_c,
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
        "a": model.weight1.item(),
        "b": model.weight2.item(),
        "c": model.bias.item(),
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
            # Store the loss corresponding to the SAME post-update parameters
            # and predictions shown in this replay frame.
            displayed_loss = loss_fn(full_predictions, y_train)

        if not torch.isfinite(displayed_loss) or not torch.isfinite(full_predictions).all():
            raise ValueError(
                f"Training diverged at epoch {epoch + 1}. "
                "Try a smaller learning rate or less aggressive optimizer settings."
            )

        history.append(
            {
                "epoch": epoch + 1,
                "a": model.weight1.item(),
                "b": model.weight2.item(),
                "c": model.bias.item(),
                "train_loss": displayed_loss.item(),
                "predictions": _to_float_list(full_predictions),
            }
        )

    final_parameters = {
        "a": model.weight1.item(),
        "b": model.weight2.item(),
        "c": model.bias.item(),
    }

    return {
        "target_parameters": {
            "a": float(true_a),
            "b": float(true_b),
            "c": float(true_c),
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


def test_quadratic(
    *,
    true_a,
    true_b,
    true_c,
    model_a,
    model_b,
    model_c,
    model_epoch=None,
    x_values=None,
):
    """
    Evaluate a trained quadratic model on NEW x-values.

    The target y-values come from the theoretically correct quadratic while
    the predicted y-values come from the learned A, B, and C. No parameters
    are updated here: this is evaluation only.
    """

    x_values = [float(value) for value in x_values]
    if not x_values:
        raise ValueError("Provide at least one x-value to test")
    if len(x_values) > 1000:
        raise ValueError("At most 1000 test x-values are allowed at once")

    _validate_finite(
        [true_a, true_b, true_c, model_a, model_b, model_c, *x_values],
        "Test inputs",
    )

    x_values = sorted(x_values)
    x = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1)

    true_y = true_a * (x ** 2) + true_b * x + true_c
    predicted_y = model_a * (x ** 2) + model_b * x + model_c
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
    target_curve_y = true_a * (curve_x ** 2) + true_b * curve_x + true_c
    prediction_curve_y = model_a * (curve_x ** 2) + model_b * curve_x + model_c

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
            "c": float(true_c),
        },
        "model_parameters": {
            "a": float(model_a),
            "b": float(model_b),
            "c": float(model_c),
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
    result = train_quadratic(epochs=500)
    print("Target:", result["target_parameters"])
    print("Final:", result["final_parameters"])
    print("Final train loss:", result["history"][-1]["train_loss"])
