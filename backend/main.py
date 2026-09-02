import random
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quadratic_backend import test_quadratic, train_quadratic
from linear_backend import test_linear, train_linear
from code_export import build_export_filename, generate_pytorch_script
from linear_code_export import build_linear_export_filename, generate_linear_pytorch_script


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Regression Playground")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


LossName = Literal["mse", "l1", "smooth_l1", "huber"]
OptimizerName = Literal["sgd", "adam", "adamw", "rmsprop", "adagrad"]


class TrainRequest(BaseModel):
    true_a: float = 3.0
    true_b: float = 1.0
    true_c: float = -1.0

    initial_a: Optional[float] = None
    initial_b: Optional[float] = None
    initial_c: Optional[float] = None
    seed: Optional[int] = None

    loss_function: LossName = "l1"
    loss_beta: float = Field(default=1.0, gt=0)
    loss_delta: float = Field(default=1.0, gt=0)

    optimizer: OptimizerName = "sgd"
    learning_rate: float = Field(default=0.01, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)

    # SGD
    momentum: float = Field(default=0.0, ge=0)
    nesterov: bool = False

    # Adam / AdamW
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    amsgrad: bool = False

    # RMSprop
    rmsprop_alpha: float = Field(default=0.99, gt=0, le=1)
    rmsprop_momentum: float = Field(default=0.0, ge=0)
    rmsprop_centered: bool = False

    # Adagrad
    adagrad_lr_decay: float = Field(default=0.0, ge=0)

    epochs: int = Field(default=1000, ge=1, le=20000)


class TestRequest(BaseModel):
    true_a: float
    true_b: float
    true_c: float

    model_a: float
    model_b: float
    model_c: float
    model_epoch: Optional[int] = Field(default=None, ge=0)

    mode: Literal["manual", "random"] = "manual"
    x_values: Optional[list[float]] = None

    random_count: int = Field(default=100, ge=1, le=1000)
    x_min: float = -1.0
    x_max: float = 2.0
    seed: Optional[int] = None


class LinearTrainRequest(BaseModel):
    true_a: float = 3.0
    true_b: float = 1.0

    initial_a: Optional[float] = None
    initial_b: Optional[float] = None
    seed: Optional[int] = None

    loss_function: LossName = "l1"
    loss_beta: float = Field(default=1.0, gt=0)
    loss_delta: float = Field(default=1.0, gt=0)

    optimizer: OptimizerName = "sgd"
    learning_rate: float = Field(default=0.01, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)

    momentum: float = Field(default=0.0, ge=0)
    nesterov: bool = False

    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    amsgrad: bool = False

    rmsprop_alpha: float = Field(default=0.99, gt=0, le=1)
    rmsprop_momentum: float = Field(default=0.0, ge=0)
    rmsprop_centered: bool = False

    adagrad_lr_decay: float = Field(default=0.0, ge=0)

    epochs: int = Field(default=1000, ge=1, le=20000)


class LinearTestRequest(BaseModel):
    true_a: float
    true_b: float

    model_a: float
    model_b: float
    model_epoch: Optional[int] = Field(default=None, ge=0)

    mode: Literal["manual", "random"] = "manual"
    x_values: Optional[list[float]] = None

    random_count: int = Field(default=100, ge=1, le=1000)
    x_min: float = -1.0
    x_max: float = 2.0
    seed: Optional[int] = None


class LinearExportCodeRequest(BaseModel):
    target_a: float
    target_b: float

    initial_a: float
    initial_b: float
    initialization_mode: Literal["manual", "random"] = "manual"
    seed: Optional[int] = None

    loss_function: LossName
    loss_beta: float = Field(default=1.0, gt=0)
    loss_delta: float = Field(default=1.0, gt=0)

    optimizer: OptimizerName
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)

    momentum: float = Field(default=0.0, ge=0)
    nesterov: bool = False

    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    amsgrad: bool = False

    rmsprop_alpha: float = Field(default=0.99, gt=0, le=1)
    rmsprop_momentum: float = Field(default=0.0, ge=0)
    rmsprop_centered: bool = False

    adagrad_lr_decay: float = Field(default=0.0, ge=0)

    epochs: int = Field(ge=1, le=20000)
    train_x: list[float]


class ExportCodeRequest(BaseModel):
    target_a: float
    target_b: float
    target_c: float

    initial_a: float
    initial_b: float
    initial_c: float
    initialization_mode: Literal["manual", "random"] = "manual"
    seed: Optional[int] = None

    loss_function: LossName
    loss_beta: float = Field(default=1.0, gt=0)
    loss_delta: float = Field(default=1.0, gt=0)

    optimizer: OptimizerName
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)

    momentum: float = Field(default=0.0, ge=0)
    nesterov: bool = False

    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    amsgrad: bool = False

    rmsprop_alpha: float = Field(default=0.99, gt=0, le=1)
    rmsprop_momentum: float = Field(default=0.0, ge=0)
    rmsprop_centered: bool = False

    adagrad_lr_decay: float = Field(default=0.0, ge=0)

    epochs: int = Field(ge=1, le=20000)
    train_x: list[float]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/linear", include_in_schema=False)
def linear_home():
    return FileResponse(FRONTEND_DIR / "linear.html")


@app.post("/api/train")
def train(request: TrainRequest):
    try:
        return train_quadratic(
            true_a=request.true_a,
            true_b=request.true_b,
            true_c=request.true_c,
            initial_a=request.initial_a,
            initial_b=request.initial_b,
            initial_c=request.initial_c,
            seed=request.seed,
            loss_function=request.loss_function,
            loss_beta=request.loss_beta,
            loss_delta=request.loss_delta,
            optimizer_name=request.optimizer,
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            momentum=request.momentum,
            nesterov=request.nesterov,
            beta1=request.beta1,
            beta2=request.beta2,
            amsgrad=request.amsgrad,
            rmsprop_alpha=request.rmsprop_alpha,
            rmsprop_momentum=request.rmsprop_momentum,
            rmsprop_centered=request.rmsprop_centered,
            adagrad_lr_decay=request.adagrad_lr_decay,
            epochs=request.epochs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/test")
def test(request: TestRequest):
    try:
        if request.mode == "manual":
            if not request.x_values:
                raise ValueError("Enter at least one x-value to test")
            x_values = request.x_values
        else:
            if request.x_max <= request.x_min:
                raise ValueError("Random x max must be greater than random x min")
            rng = random.Random(request.seed)
            x_values = [
                rng.uniform(request.x_min, request.x_max)
                for _ in range(request.random_count)
            ]

        return test_quadratic(
            true_a=request.true_a,
            true_b=request.true_b,
            true_c=request.true_c,
            model_a=request.model_a,
            model_b=request.model_b,
            model_c=request.model_c,
            model_epoch=request.model_epoch,
            x_values=x_values,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/export/code")
def export_code(request: ExportCodeRequest):
    try:
        if len(request.train_x) < 2:
            raise ValueError("At least two training x-values are required for export")
        if len(request.train_x) > 5000:
            raise ValueError("Too many training x-values to export")

        code = generate_pytorch_script(
            target_a=request.target_a,
            target_b=request.target_b,
            target_c=request.target_c,
            initial_a=request.initial_a,
            initial_b=request.initial_b,
            initial_c=request.initial_c,
            initialization_mode=request.initialization_mode,
            seed=request.seed,
            loss_function=request.loss_function,
            loss_beta=request.loss_beta,
            loss_delta=request.loss_delta,
            optimizer_name=request.optimizer,
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            momentum=request.momentum,
            nesterov=request.nesterov,
            beta1=request.beta1,
            beta2=request.beta2,
            amsgrad=request.amsgrad,
            rmsprop_alpha=request.rmsprop_alpha,
            rmsprop_momentum=request.rmsprop_momentum,
            rmsprop_centered=request.rmsprop_centered,
            adagrad_lr_decay=request.adagrad_lr_decay,
            epochs=request.epochs,
            train_x=request.train_x,
        )

        return {
            "filename": build_export_filename(
                optimizer_name=request.optimizer,
                loss_function=request.loss_function,
                epochs=request.epochs,
            ),
            "code": code,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc



@app.post("/api/linear/train")
def train_linear_endpoint(request: LinearTrainRequest):
    try:
        return train_linear(
            true_a=request.true_a,
            true_b=request.true_b,
            initial_a=request.initial_a,
            initial_b=request.initial_b,
            seed=request.seed,
            loss_function=request.loss_function,
            loss_beta=request.loss_beta,
            loss_delta=request.loss_delta,
            optimizer_name=request.optimizer,
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            momentum=request.momentum,
            nesterov=request.nesterov,
            beta1=request.beta1,
            beta2=request.beta2,
            amsgrad=request.amsgrad,
            rmsprop_alpha=request.rmsprop_alpha,
            rmsprop_momentum=request.rmsprop_momentum,
            rmsprop_centered=request.rmsprop_centered,
            adagrad_lr_decay=request.adagrad_lr_decay,
            epochs=request.epochs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/linear/test")
def test_linear_endpoint(request: LinearTestRequest):
    try:
        if request.mode == "manual":
            if not request.x_values:
                raise ValueError("Enter at least one x-value to test")
            x_values = request.x_values
        else:
            if request.x_max <= request.x_min:
                raise ValueError("Random x max must be greater than random x min")
            rng = random.Random(request.seed)
            x_values = [
                rng.uniform(request.x_min, request.x_max)
                for _ in range(request.random_count)
            ]

        return test_linear(
            true_a=request.true_a,
            true_b=request.true_b,
            model_a=request.model_a,
            model_b=request.model_b,
            model_epoch=request.model_epoch,
            x_values=x_values,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/linear/export/code")
def export_linear_code(request: LinearExportCodeRequest):
    try:
        if len(request.train_x) < 2:
            raise ValueError("At least two training x-values are required for export")
        if len(request.train_x) > 5000:
            raise ValueError("Too many training x-values to export")

        code = generate_linear_pytorch_script(
            target_a=request.target_a,
            target_b=request.target_b,
            initial_a=request.initial_a,
            initial_b=request.initial_b,
            initialization_mode=request.initialization_mode,
            seed=request.seed,
            loss_function=request.loss_function,
            loss_beta=request.loss_beta,
            loss_delta=request.loss_delta,
            optimizer_name=request.optimizer,
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            momentum=request.momentum,
            nesterov=request.nesterov,
            beta1=request.beta1,
            beta2=request.beta2,
            amsgrad=request.amsgrad,
            rmsprop_alpha=request.rmsprop_alpha,
            rmsprop_momentum=request.rmsprop_momentum,
            rmsprop_centered=request.rmsprop_centered,
            adagrad_lr_decay=request.adagrad_lr_decay,
            epochs=request.epochs,
            train_x=request.train_x,
        )

        return {
            "filename": build_linear_export_filename(
                optimizer_name=request.optimizer,
                loss_function=request.loss_function,
                epochs=request.epochs,
            ),
            "code": code,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
