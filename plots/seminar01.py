"""Visual style and computational-graph helpers for seminar 01."""

from pathlib import Path

from IPython.display import SVG, display
import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    "data": "#dff3ef",
    "parameter": "#dce8f7",
    "prediction": "#fff0c7",
    "loss": "#f9d9dc",
    "operation": "#f2f2f2",
    "forward": "#287c73",
    "backward": "#bb4d5b",
    "border": "#4e6662",
    "text": "#202124",
}


def configure_plots():
    plt.rcParams.update(
        {
            "font.size": 12,
            "figure.figsize": (9, 4.5),
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def box(ax, position, label, kind="operation", size=11):
    ax.text(
        *position,
        label,
        ha="center",
        va="center",
        fontsize=size,
        color=COLORS["text"],
        bbox={
            "boxstyle": "round,pad=0.5,rounding_size=0.15",
            "facecolor": COLORS[kind],
            "edgecolor": COLORS["border"],
            "lw": 1.25,
        },
    )


def arrow(ax, start, end, label="", color=None, radius=0.0, shrink=35):
    color = color or COLORS["forward"]
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 2,
            "mutation_scale": 13,
            "shrinkA": shrink,
            "shrinkB": shrink,
            "connectionstyle": f"arc3,rad={radius}",
        },
    )
    if label:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2 + (0.13 if radius >= 0 else -0.15)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.5},
        )


GRAPH_EDGES = [
    ('x', 'm'), ('w', 'm'), ('m', 'yhat'), ('b', 'yhat'),
    ('yhat', 'e'), ('y', 'e'), ('e', 'loss'),
]


def plot_bce(p):
    """Show odds, logits/sigmoid, and BCE on one compact canvas.

    The first two rows follow Godoy's Chapter 3, figure2 and figure3:
    https://github.com/dvgodoy/PyTorchStepByStep/blob/master/plots/chapter3.py
    The final true-class-probability plot is a course adaptation.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size < 2 or not np.all(np.isfinite(p) & (p > 0) & (p < 1)):
        raise ValueError('Передайте одномерную сетку вероятностей строго между 0 и 1.')
    p = np.sort(p)
    marks = np.array([.25, .5, .75])
    odds = p / (1 - p)
    mark_odds = marks / (1 - marks)
    logits = np.log(odds)
    mark_logits = np.log(mark_odds)
    curve_color, mark_color = '#287c73', '#bb4d5b'

    fig = plt.figure(figsize=(9.5, 6.3), dpi=100, layout='constrained')
    grid = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    for ax in axes:
        ax.plot(p, odds, color=curve_color, lw=2)
        ax.scatter(marks, mark_odds, color=mark_color, zorder=3)
        ax.set(xlabel='Вероятность класса 1, p', xticks=marks, xlim=(0, 1))
    axes[0].set(title='1. Шансы', ylabel=r'$p/(1-p)$', ylim=(0, 10))
    axes[1].set(yscale='log', title='1. Те же шансы: логарифмическая шкала',
                ylabel=r'$p/(1-p)$', yticks=[1/3, 1, 3], yticklabels=['1/3', '1', '3'])
    axes = [fig.add_subplot(grid[1, column]) for column in range(2)]
    axes[0].plot(p, logits, color=curve_color, lw=2)
    axes[0].scatter(marks, mark_logits, color=mark_color, zorder=3)
    axes[0].set(title='2. Логарифм шансов', xlabel='Вероятность класса 1, p',
                ylabel=r'$z=\log\frac{p}{1-p}$', xticks=marks, xlim=(0, 1))
    axes[1].plot(logits, p, color=curve_color, lw=2)
    axes[1].scatter(mark_logits, marks, color=mark_color, zorder=3)
    axes[1].set(title='2. Обратная зависимость: сигмоида', xlabel='Логит z',
                ylabel=r'Вероятность $p=\sigma(z)$', yticks=marks, ylim=(0, 1))
    axes[0].axhline(0, color='#606060', ls='--', lw=1)
    axes[1].axvline(0, color='#606060', ls='--', lw=1)
    q = p
    ax = fig.add_subplot(grid[2, :])
    ax.plot(q, -np.log(q), color=curve_color, lw=2.5, label=r'$\ell=-\log q$')
    ax.scatter([.1, .9], -np.log([.1, .9]), color=mark_color, zorder=3)
    ax.annotate('y = 0: q = 1 - p = 0.1', (.1, -np.log(.1)), xytext=(.23, 3.2),
                fontsize=9, arrowprops=dict(arrowstyle='->', color=mark_color))
    ax.annotate('y = 1: q = p = 0.9', (.9, -np.log(.9)), xytext=(.53, 1.3),
                fontsize=9, arrowprops=dict(arrowstyle='->', color=mark_color))
    ax.set(xlabel='Вероятность, присвоенная истинному классу, q',
           ylabel='Потеря', xlim=(0, 1), ylim=(-.05, 5),
           title='3. Потеря: модель оценила вероятность класса 1 как p = 0.9')
    ax.legend(loc='upper right', fontsize=9)
    for panel in fig.axes:
        panel.title.set_fontsize(10)
        panel.xaxis.label.set_fontsize(9)
        panel.yaxis.label.set_fontsize(9)
        panel.tick_params(labelsize=8)
    plt.show()


def plot_outlier_influence(outlier_distance, mse_optimum, mae_optimum, mse_influence, mae_influence):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')

    axes[0].loglog(outlier_distance, mse_optimum, color='#bb4d5b', lw=2.5,
                   label='Минимум MSE: среднее')
    axes[0].loglog(outlier_distance, mae_optimum, color='#287c73', lw=2.5,
                   label='Минимум MAE: медиана')
    axes[0].set(xlabel='Положение выброса', ylabel='Оптимальная константа',
                title='Куда смещается оптимум?')
    axes[0].legend()

    axes[1].loglog(outlier_distance, mse_influence, color='#bb4d5b', lw=2.5,
                   label=r'MSE: $2|c-y|$')
    axes[1].loglog(outlier_distance, mae_influence, color='#287c73', lw=2.5,
                   label=r'MAE: $|\partial |c-y|/\partial c|=1$')
    axes[1].annotate('Всё ещё один голос', (1e6, 1), xytext=(2e3, 15),
                     arrowprops=dict(arrowstyle='->', color='#287c73'),
                     color='#287c73', weight='bold')
    axes[1].set(xlabel='Положение выброса', ylabel='Модуль вклада в производную',
                title='Насколько сильно влияет выброс?')
    axes[1].legend()

    fig.suptitle('MAE: влияние выброса ограничено; MSE: растёт с отклонением')
    plt.show()


def plot_data(x_train, y_train, x_validation, y_validation, true_b, true_w):
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    ax.scatter(x_train, y_train, s=24, color='#287c73', label='Обучение')
    ax.scatter(x_validation, y_validation, s=30, marker='x', color='#bb4d5b', label='Проверка')
    grid = np.linspace(0, 1, 100)
    ax.plot(grid, true_b+true_w*grid, color='#202124', lw=2, label='Исходная зависимость')
    ax.set(xlabel='x', ylabel='y', title='Данные для линейной регрессии')
    ax.legend()
    plt.show()


def plot_initial_model(x_train, y_train, true_b, true_w, parameters_initial, predict_numpy, loss_initial):
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    ax.scatter(x_train, y_train, s=24, color='#287c73', label='Обучение')
    grid = np.linspace(0, 1, 100)
    ax.plot(grid, true_b+true_w*grid, '--', color='#202124', lw=2, label='Исходная зависимость')
    ax.plot(grid, predict_numpy(grid, parameters_initial), color='#bb4d5b', lw=2.5,
            label='Случайная инициализация')
    ax.set(xlabel='x', ylabel='y', title=f'Предсказания до обучения: MSE = {loss_initial:.3f}')
    ax.legend()
    plt.show()


def plot_loss_landscape(bs, ws, losses_grid, parameters_initial, optimal_parameters, b_range, w_range, loss_by_b, loss_by_w, b_initial, w_initial):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    contours = axes[0].contour(bs, ws, losses_grid, levels=14, cmap='viridis')
    axes[0].clabel(contours, fontsize=8)
    axes[0].scatter(*parameters_initial, color='#bb4d5b', s=50, label='Начальная точка')
    axes[0].scatter(*optimal_parameters, color='#287c73', s=50, label='Минимум')
    axes[0].set(xlabel='b', ylabel='w', title='Линии уровня потерь')
    axes[0].legend()

    axes[1].plot(b_range, loss_by_b, color='#bb4d5b', lw=2.5)
    axes[1].axvline(b_initial, color='#202124', ls='--', label=f'b = {b_initial:.2f}')
    axes[1].set(xlabel='b (w фиксирован)', ylabel='MSE', title='Меняем b, фиксируем w')
    axes[1].legend()

    axes[2].plot(w_range, loss_by_w, color='#287c73', lw=2.5)
    axes[2].axvline(w_initial, color='#202124', ls='--', label=f'w = {w_initial:.2f}')
    axes[2].set(xlabel='w (b фиксирован)', ylabel='MSE', title='Меняем w, фиксируем b')
    axes[2].legend()
    plt.show()


def plot_difference_error(steps, forward_errors, central_errors):
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    ax.loglog(steps, np.maximum(forward_errors, 1e-16), 'o-', color='#287c73', label='Правая разность')
    ax.loglog(steps, np.maximum(central_errors, 1e-16), 'o-', color='#bb4d5b', label='Центральная разность')
    ax.set(xlabel='h', ylabel=r'$\|g_{числ}-g_{аналит}\|_2$',
           title='При слишком малом h мешают ошибки округления')
    ax.legend()
    plt.show()


def plot_learning_rates(bs, ws, losses_grid, parameters_initial, analytic_gradient, shown_rates, learning_rates, loss_after_step, loss_initial, mse_numpy):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    contours = axes[0].contour(bs, ws, losses_grid, levels=16, cmap='viridis')
    axes[0].scatter(*parameters_initial, color='#202124', s=45, zorder=4)
    for lr, color in zip(shown_rates, ['#287c73', '#d08c00', '#bb4d5b']):
        updated = parameters_initial-lr*analytic_gradient
        axes[0].annotate('', xy=updated, xytext=parameters_initial,
                         arrowprops=dict(arrowstyle='-|>', color=color, lw=2))
        axes[0].scatter(*updated, color=color, s=35, label=f'η={lr:g}')
    axes[0].set(xlabel='b', ylabel='w', title='Один шаг на поверхности потерь')
    axes[0].legend()

    axes[1].semilogx(learning_rates, loss_after_step, color='#287c73', lw=2.5)
    axes[1].axhline(loss_initial, color='#202124', ls='--', label='Потеря до обновления')
    for lr, color in zip(shown_rates, ['#287c73', '#d08c00', '#bb4d5b']):
        axes[1].scatter(lr, mse_numpy(parameters_initial-lr*analytic_gradient), color=color, s=45)
    axes[1].set(xlabel='Темп обучения', ylabel='MSE после одного шага',
                title='Одно направление, разная длина шага')
    axes[1].legend()
    plt.show()


def plot_graph_structure():
    graph_positions = {
        'x': ((0.0, 1.45), r'$x$', 'data'),
        'w': ((0.0, .65), r'$w$', 'parameter'),
        'b': ((1.9, -.05), r'$b$', 'parameter'),
        'm': ((1.9, 1.05), r'$m=wx$', 'operation'),
        'yhat': ((3.55, .75), r'$\hat y=b+m$', 'prediction'),
        'y': ((3.55, -.2), r'$y$', 'data'),
        'e': ((5.15, .55), r'$e=\hat y-y$', 'operation'),
        'loss': ((6.65, .55), r'$\ell=e^2$', 'loss'),
    }
    fig, ax = plt.subplots(figsize=(13, 4.3), layout='constrained')
    for _, (position, label, kind) in graph_positions.items():
        box(ax, position, label, kind)
    for start, end in GRAPH_EDGES:
        arrow(ax, graph_positions[start][0], graph_positions[end][0], shrink=38)
    ax.set(xlim=(-.6, 7.25), ylim=(-.55, 1.85),
           title='Структура вычислительного графа')
    ax.axis('off')
    plt.show()


def plot_graph_forward():
    forward_nodes = {
        'x': ((0.0, 1.45), r'$x=0.5$', 'data'),
        'w': ((0.0, .65), r'$w=1$', 'parameter'),
        'b': ((1.9, -.05), r'$b=0.5$', 'parameter'),
        'm': ((1.9, 1.05), r'$m=0.5$', 'operation'),
        'yhat': ((3.55, .75), r'$\hat y=1$', 'prediction'),
        'y': ((3.55, -.2), r'$y=2$', 'data'),
        'e': ((5.15, .55), r'$e=-1$', 'operation'),
        'loss': ((6.65, .55), r'$\ell=1$', 'loss'),
    }
    fig, ax = plt.subplots(figsize=(13, 4.3), layout='constrained')
    for _, (position, label, kind) in forward_nodes.items():
        box(ax, position, label, kind)
    for start, end in GRAPH_EDGES:
        arrow(ax, forward_nodes[start][0], forward_nodes[end][0], shrink=38)
    ax.set(xlim=(-.6, 7.25), ylim=(-.55, 1.85), title='Прямой проход: промежуточные значения')
    ax.axis('off')
    plt.show()


def plot_graph_backward():
    backward_nodes = {
        'x': ((0.0, 1.45), r'$\bar x=-2$', 'data'),
        'w': ((0.0, .65), r'$\bar w=-1$', 'parameter'),
        'b': ((1.9, -.05), r'$\bar b=-2$', 'parameter'),
        'm': ((1.9, 1.05), r'$\bar m=-2$', 'operation'),
        'yhat': ((3.55, .75), r'$\bar{\hat y}=-2$', 'prediction'),
        'y': ((3.55, -.2), r'$\bar y=2$', 'data'),
        'e': ((5.15, .55), r'$\bar e=2e=-2$', 'operation'),
        'loss': ((6.65, .55), r'$\bar\ell=1$', 'loss'),
    }
    fig, ax = plt.subplots(figsize=(13, 4.3), layout='constrained')
    for _, (position, label, kind) in backward_nodes.items():
        box(ax, position, label, kind)
    for start, end in GRAPH_EDGES:
        arrow(ax, backward_nodes[end][0], backward_nodes[start][0],
              color=COLORS['backward'], shrink=42)
    ax.set(xlim=(-.6, 7.25), ylim=(-.55, 1.85),
           title=r'Обратный проход: $\bar v=\partial\ell/\partial v$')
    ax.axis('off')
    plt.show()


def plot_batch_contributions(x_train, per_item_b, per_item_w, order):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    axes[0].bar(np.arange(x_train.size), per_item_b[order], color='#287c73')
    axes[0].axhline(per_item_b.mean(), color='#bb4d5b', lw=2,
                    label=f'Среднее = {per_item_b.mean():.3f}')
    axes[0].set(xlabel='Объекты обучения по возрастанию x', ylabel='Вклад',
                title=r'Вклады $2e_i$ в $\partial L/\partial b$')
    axes[0].legend()

    axes[1].bar(np.arange(x_train.size), per_item_w[order], color='#287c73')
    axes[1].axhline(per_item_w.mean(), color='#bb4d5b', lw=2,
                    label=f'Среднее = {per_item_w.mean():.3f}')
    axes[1].set(xlabel='Объекты обучения по возрастанию x', ylabel='Вклад',
                title=r'Вклады $2x_ie_i$ в $\partial L/\partial w$')
    axes[1].legend()
    plt.show()


def plot_bce_gradient(p_grid):
    fig, ax = plt.subplots(figsize=(10, 4.6), layout='constrained')
    ax.plot(p_grid, p_grid-1, color='#287c73', lw=2.5, label=r'$y=1:\ p-y$')
    ax.plot(p_grid, p_grid, color='#bb4d5b', lw=2.5, label=r'$y=0:\ p-y$')
    ax.axhline(0, color='#202124', lw=1)
    ax.scatter([.9, .2], [-.1, .2], color=['#287c73', '#bb4d5b'], zorder=3)
    ax.set(xlabel='p', ylabel=r'$\partial\ell/\partial z$',
           title='Обратный сигнал: сигмоида и BCE')
    ax.legend()
    plt.show()


def plot_update(x_train, y_train, true_b, true_w, parameters_initial, parameters_updated, predict_numpy, loss_initial, loss_updated):
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    ax.scatter(x_train, y_train, s=22, color='#287c73', label='Обучение')
    grid = np.linspace(0, 1, 100)
    ax.plot(grid, predict_numpy(grid, parameters_initial), color='#bb4d5b', lw=2,
            label=f'До шага: потеря={loss_initial:.3f}')
    ax.plot(grid, predict_numpy(grid, parameters_updated), color='#d08c00', lw=2.5,
            label=f'После шага: потеря={loss_updated:.3f}')
    ax.plot(grid, true_b+true_w*grid, '--', color='#202124', lw=1.5, label='Исходная зависимость')
    ax.set(xlabel='x', ylabel='y', title='Прямой проход → обратный проход → обновление')
    ax.legend()
    plt.show()


def plot_grades(total_points, five, six):
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.step(total_points, five, where="mid", label="5 домашних заданий", color=COLORS["forward"])
    ax.step(total_points, six, where="mid", label="6 домашних заданий", color=COLORS["backward"])
    ax.axhline(3, color=COLORS["text"], ls="--", label="Порог зачёта")
    ax.set(xlabel="Сумма баллов за домашние задания", ylabel="Оценка",
           title="Два сценария расчёта оценки", yticks=range(11), ylim=(-0.3, 10.5))
    ax.legend()
    plt.show()


def plot_training_loop():
    display(SVG(filename=str(Path(__file__).parent / "assets" / "training_loop.svg")))
