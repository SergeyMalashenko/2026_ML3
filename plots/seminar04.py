"""Compact convolution figures. Godoy's numeric example: see LICENSE-GODOY."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_plots():
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("retina")
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9,
        "figure.figsize": (8, 3), "figure.dpi": 100,
        "savefig.dpi": "figure", "axes.grid": False,
    })


def image_path():
    return Path(__file__).parent / "assets" / "seminar04-volleyball.jpeg"


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def show_images(images, titles, *, signed=False):
    count = len(images)
    cols = min(count, 4)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(min(10, 2.5 * cols), 2.35 * rows),
                             squeeze=False)
    values = [array(image).squeeze() for image in images]
    limit = max(float(np.abs(image).max()) for image in values) or 1
    for ax, value, title in zip(axes.flat, values, titles):
        if value.ndim == 3 and value.shape[0] == 3:
            value = value.transpose(1, 2, 0)
        kwargs = {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit} if signed else {"cmap": "gray"}
        ax.imshow(value, **kwargs)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.flat[count:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_window(image, kernel, row=0, col=0):
    image, kernel = array(image).squeeze(), array(kernel).squeeze()
    patch = image[row:row+3, col:col+3]
    values = [image, kernel, patch * kernel]
    titles = [f"Изображение: окно ({row}, {col})", "Общее ядро", f"Произведения, сумма = {(patch*kernel).sum():g}"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, value, title in zip(axes, values, titles):
        ax.imshow(value, cmap="Blues", vmin=-1, vmax=max(9, value.max()))
        for (i, j), number in np.ndenumerate(value):
            color = "white" if number > 5 else "#111111"
            ax.text(j, i, f"{number:g}", ha="center", va="center", color=color, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    from matplotlib.patches import Rectangle
    axes[0].add_patch(Rectangle((col-.5, row-.5), 3, 3, fill=False, edgecolor="#bb4d5b", lw=2))
    fig.tight_layout()
    return fig


def plot_training(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, color in [("MLP", "#287c73"), ("CNN", "#bb4d5b")]:
        selected = [r for r in runs if r["model"] == name]
        for field, style, label in [("train_loss", "--", "обучающая"), ("val_loss", "-", "валидационная")]:
            ys = np.array([[e[field] for e in r["history"]] for r in selected])
            xs = np.arange(1, ys.shape[1] + 1)
            for y in ys:
                axes[0].plot(xs, y, color=color, ls=style, alpha=.18)
            axes[0].plot(xs, ys.mean(0), color=color, ls=style, label=f"{name}: {label}")
        ys = np.array([[e["val_accuracy"] for e in r["history"]] for r in selected]) * 100
        for y in ys:
            axes[1].plot(xs, y, color=color, alpha=.25)
        axes[1].plot(xs, ys.mean(0), color=color, label=name)
    for ax in axes:
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Средняя кросс-энтропия (eval)")
    axes[1].set_ylabel("Верных ответов на валидации, %")
    fig.tight_layout()
    return fig


def plot_shifts(results):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, subset, title in zip(axes, ["all", "interior"], ["Вся валидация (возможна обрезка)", "Общая подвыборка без обрезки"]):
        for name, color in [("MLP", "#287c73"), ("CNN", "#bb4d5b")]:
            seeds = sorted({r["seed"] for r in results})
            ys = []
            for seed in seeds:
                row = []
                for distance in [0, 2, 4]:
                    values = [r["accuracy"] for r in results if r["model"] == name and r["seed"] == seed
                              and r["subset"] == subset and r["distance"] == distance]
                    row.append(np.mean(values))
                ys.append(row)
            ys = np.asarray(ys) * 100
            for y in ys:
                ax.plot([0, 2, 4], y, color=color, alpha=.25)
            ax.plot([0, 2, 4], ys.mean(0), "o-", color=color, label=name)
        ax.set(title=title, xlabel="Сдвиг, пиксели", ylabel="Верных ответов, %", xticks=[0, 2, 4])
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    return fig
