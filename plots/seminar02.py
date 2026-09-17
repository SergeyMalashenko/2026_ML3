"""Compact figures for seminar 2. Godoy adaptations retain the figure1/3 API."""

import matplotlib.pyplot as plt
import numpy as np


def configure_plots():
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("retina")
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9,
        "figure.figsize": (7.5, 3.5), "figure.dpi": 100,
        "savefig.dpi": "figure", "axes.grid": True, "grid.alpha": 0.22,
    })


# Adapted from Daniel Voigt Godoy, plots/chapter1.py (MIT; see LICENSE-GODOY).
def fit_model(x_train, y_train):
    from sklearn.linear_model import LinearRegression

    regression = LinearRegression().fit(x_train, y_train)
    return regression.intercept_[0], regression.coef_[0][0]


def figure1(x_train, y_train, x_val, y_val):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].scatter(x_train, y_train)
    ax[0].set(xlabel="x", ylabel="y", ylim=(0, 3.1), title="Обучающая выборка")
    ax[1].scatter(x_val, y_val, c="r")
    ax[1].set(xlabel="x", ylabel="y", ylim=(0, 3.1), title="Валидационная выборка")
    fig.tight_layout()
    return fig, ax


def figure3(x_train, y_train):
    b_minimum, w_minimum = fit_model(x_train, y_train)
    x_range = np.linspace(0, 1, 101)
    yhat_range = b_minimum + w_minimum * x_range
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.set(xlabel="x", ylabel="y", ylim=(0, 3.1))
    ax.scatter(x_train, y_train)
    ax.plot(x_range, yhat_range, label="Решение методом наименьших квадратов", c="k", linestyle="--")
    ax.annotate(f"b = {b_minimum:.4f}, w = {w_minimum:.4f}", xy=(.3, 1.35), rotation=25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_data(x_train, y_train, x_validation, y_validation):
    fig, ax = plt.subplots(layout="constrained")
    ax.scatter(x_train, y_train, s=15, color="#287c73", label="Обучающая выборка")
    ax.scatter(x_validation, y_validation, s=22, marker="x", color="#bb4d5b", label="Валидационная выборка")
    ax.plot([0, 1], [1, 3], color="#465980", ls="--", label="Зависимость без шума")
    ax.set(xlabel="Признак x", ylabel="Отклик y", title="Те же данные, что на первом семинаре")
    ax.legend()
    plt.show()
    return fig


def plot_shapes():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3), layout="constrained")
    for ax, values, title in zip(axes, [np.array([[0], [1], [-1]]), np.array([[0, -1, -3], [2, 1, -1], [2, 1, -1]])],
                                 ["(3, 1) − (3, 1): три ошибки", "(3, 1) − (3,): девять разностей"]):
        ax.imshow(values, cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
        for (row, col), value in np.ndenumerate(values):
            ax.text(col, row, str(value), ha="center", va="center", color="#202124")
        ax.set_title(title)
        ax.set_xlabel("Столбец")
        ax.set_ylabel("Объект")
        ax.set_xticks(range(values.shape[1]))
        ax.set_yticks(range(values.shape[0]))
    plt.show()
    return fig


def plot_training(x_train, y_train, x_validation, y_validation, parameters, history):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    x_line = np.linspace(0, 1, 100)
    axes[0].scatter(x_train, y_train, s=12, color="#287c73", label="Обучение")
    axes[0].scatter(x_validation, y_validation, s=20, marker="x", color="#bb4d5b", label="Валидация")
    axes[0].plot(x_line, parameters[0] + parameters[1] * x_line, color="#465980", label="Предсказание")
    axes[0].set(xlabel="Признак x", ylabel="Отклик y", title="Результат обучения")
    axes[0].legend()
    steps = np.arange(len(history))
    axes[1].plot(steps, np.asarray(history)[:, 0], label="Обучение", color="#287c73")
    axes[1].plot(steps, np.asarray(history)[:, 1], label="Валидация", color="#bb4d5b")
    axes[1].set(xlabel="Число обновлений", ylabel="Средний квадрат ошибки", yscale="log", title="Оценка при одних и тех же весах")
    axes[1].legend()
    plt.show()
    return fig
