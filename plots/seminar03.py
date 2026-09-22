"""Chapter 2 plots adapted from D. V. Godoy (MIT; see LICENSE-GODOY)."""

import matplotlib.pyplot as plt


def configure_plots():
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("retina")
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9,
        "figure.figsize": (7.5, 3.2), "figure.dpi": 100,
        "savefig.dpi": "figure", "axes.grid": True, "grid.alpha": 0.22,
    })


def plot_losses(losses, val_losses):
    fig, ax = plt.subplots()
    ax.plot(losses, label="Обучение: потери по ходу эпохи", color="#287c73")
    ax.plot(val_losses, label="Валидация: после эпохи", color="#bb4d5b")
    ax.set(yscale="log", xlabel="Индекс эпохи", ylabel="MSE")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_resumed_losses(saved_epoch, saved_losses, saved_val_losses,
                        n_epochs, losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    before = range(saved_epoch)
    after = range(saved_epoch, saved_epoch + n_epochs)
    ax.plot(before, saved_losses, "--", color="#287c73", label="Обучение до сохранения")
    ax.plot(before, saved_val_losses, "--", color="#bb4d5b", label="Валидация до сохранения")
    ax.plot(after, losses, color="#287c73", label="Обучение после загрузки")
    ax.plot(after, val_losses, color="#bb4d5b", label="Валидация после загрузки")
    ax.axvline(saved_epoch, color="#555555", linestyle=":", label="Возобновление")
    ax.set(yscale="log", xlabel="Индекс эпохи", ylabel="MSE")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_data(x_train, y_train, x_val, y_val):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.scatter(x_train, y_train, s=14, label="Обучающая выборка", color="#287c73")
    ax.scatter(x_val, y_val, s=20, marker="x", label="Валидационная выборка", color="#bb4d5b")
    ax.set(xlabel="Признак x", ylabel="Отклик y")
    ax.legend()
    fig.tight_layout()
    return fig
