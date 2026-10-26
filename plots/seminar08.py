"""Sequence figures; Godoy's generator is preserved under LICENSE-GODOY."""

import matplotlib.pyplot as plt
import numpy as np


def configure_plots():
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("retina")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "figure.figsize": (8, 3), "figure.dpi": 100,
                         "savefig.dpi": "figure", "axes.grid": False})


def generate_sequences(n=128, variable_len=False, seed=13):
    # Unchanged data-generation algorithm from Godoy, Chapter 8.
    basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    np.random.seed(seed)
    bases = np.random.randint(4, size=n)
    if variable_len:
        lengths = np.random.randint(3, size=n) + 2
    else:
        lengths = [4] * n
    directions = np.random.randint(2, size=n)
    points = [basic_corners[[(b + i) % 4 for i in range(4)]][slice(None, None, d*2-1)][:l] + np.random.randn(l, 2) * 0.1 for b, d, l in zip(bases, directions, lengths)]
    return points, directions


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def plot_sequences(points, directions):
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    for ax, sequence, direction in zip(axes, points[:3], directions[:3]):
        sequence = array(sequence)
        ax.plot(*sequence.T, "o-", color="#287c73")
        for i, (x, y) in enumerate(sequence):
            ax.annotate(str(i + 1), (x, y), xytext=(5, 5), textcoords="offset points")
        ax.set(title=f"Метка направления: {direction}", xlabel="Первая координата",
               ylabel="Вторая координата", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_states(states):
    values = array(states).reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for k, color in enumerate(["#287c73", "#bb4d5b"]):
        ax.plot(range(len(values)), values[:, k], "o-", color=color,
                label=f"Компонента {k + 1}")
    ax.set(xlabel="Число обработанных точек", ylabel="Значение состояния",
           title="Одни веса, меняющееся состояние", xticks=range(len(values)))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_feature_sequence(features):
    values = array(features)[0, :, 0, :]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set(xlabel="Позиция по ширине = шаг последовательности",
           ylabel="Канал", title="Карта признаков: один столбец = один вектор")
    fig.tight_layout()
    return fig


def plot_ctc(probabilities, labels):
    values = array(probabilities).T
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                color="white" if value > .55 else "black")
    ax.set(yticks=range(len(labels)), yticklabels=labels,
           xticks=range(values.shape[1]), xticklabels=range(1, values.shape[1]+1),
           xlabel="Шаг", title="Вероятности символов, включая blank")
    fig.tight_layout()
    return fig


def plot_stn(before, after):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.4))
    for ax, tensor, title in zip(axes, [before, after], ["Вход", "STN: начальное преобразование"]):
        value = array(tensor)[0].transpose(1, 2, 0)
        ax.imshow(np.clip(value, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_ocr_samples(images, references, predictions=None):
    values = array(images)
    mean = np.array([.485, .456, .406])
    std = np.array([.229, .224, .225])
    count = min(8, len(values))
    fig, axes = plt.subplots((count+1)//2, 2, figsize=(10, 1.6*((count+1)//2)), squeeze=False)
    for index, ax in enumerate(axes.flat):
        ax.axis('off')
        if index >= count:
            continue
        rgb = values[index].transpose(1, 2, 0)*std+mean
        ax.imshow(np.clip(rgb, 0, 1))
        title = 'Разметка: '+references[index]
        if predictions is not None:
            title += '\nРаспознано: '+(predictions[index] or '(пустая строка)')
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def _localized_godoy(name, *args, **kwargs):
    """Keep Godoy's computations/composition; translate text and bound display size."""
    from . import _godoy08
    from matplotlib.text import Text
    if name == 'hidden_states_contour':
        args = (args[0], np.asarray(args[1]), *args[2:])
    fig = getattr(_godoy08, name)(*args, **kwargs)
    replacements = [
        (' Counter-\nClockwise', 'Против часовой\nстрелки'),
        ('Counter-Clockwise', 'Против часовой'), ('Clockwise', 'По часовой'),
        ('Hidden State #', 'Состояние, шаг '), ('Input #', 'Вход '),
        ('Initial', 'Начальное'), ('Start at', 'Начало в'), ('Start\n at', 'Начало\nв'),
        ('Epoch:', 'Эпоха:'), ('hidden\\ state', 'состояние'),
        ('transformed\\ state', 'преобразованное\\ состояние'),
        ('activated\\ state', 'после\\ tanh'), ('adding\\', 'сложение\\'),
        ('reset\\ gate', 'гейт\\ сброса'), ('update\\ gate', 'гейт\\ обновления'),
    ]
    for artist in fig.findobj(match=Text):
        text = artist.get_text()
        for english, russian in replacements:
            text = text.replace(english, russian)
        artist.set_text(text)
        artist.set_fontsize(min(artist.get_fontsize(), 9))
    if name == 'hidden_states_contour':
        fig.set_size_inches(9, 6.8)
        panels = [ax for ax in fig.axes if ax.get_xlabel()]
        colorbars = [ax for ax in fig.axes if not ax.get_xlabel()]
        for i, ax in enumerate(panels):
            row, column = divmod(i, 2)
            ax.set_position([.1+column*.43, .59-row*.48, .32, .32])
            ax.yaxis.set_label_coords(-.19, .5)
            ax.tick_params(labelsize=8)
        for row, ax in enumerate(colorbars):
            ax.set_position([.9, .59-row*.48, .015, .32])
            ax.tick_params(labelsize=8)
        return fig
    if name in ('figure20', 'figure25'):
        panels = [ax for ax in fig.axes if ax.get_xlabel()]
        colorbars = [ax for ax in fig.axes if not ax.get_xlabel()]
        count = len(panels)
        fig.set_size_inches(12 if count == 3 else 9, 3.8)
        for i, ax in enumerate(panels):
            ax.set_position([.07+i*.86/count, .18, .86/count-.065, .62])
            ax.yaxis.set_label_coords(-.21, .5)
            ax.tick_params(labelsize=8)
        for ax in colorbars:
            ax.set_position([.94, .18, .012, .62])
            ax.tick_params(labelsize=8)
        return fig
    if name in ('figure16', 'figure22'):
        # The original four-row trajectories are shown one time step at a time.
        # Rendering separate rows keeps every original panel without tiny text.
        from io import BytesIO
        from IPython.display import Image, display
        from matplotlib.transforms import Bbox
        columns = len(fig.axes) // 4
        titles = ([r'Состояние $h$', r'Линейная часть $t_h$', r'Сумма $t_h+t_x$', r'Новое состояние $h$']
                  if columns == 4 else
                  [r'Состояние $h$', r'Линейная часть $t_h$', r'Сброс $r\odot t_h$',
                   r'Сумма с $t_x$', r'Кандидат $n$', r'Обновление $(1-z)n$', r'Новое состояние $h$'])
        fig.set_size_inches(12, 11)
        for index, ax in enumerate(fig.axes):
            row, column = divmod(index, columns)
            ax.set_position([.07 + column * (.91/columns), .79-row*.245,
                             .91/columns-.035, .145])
            ax.set_title(titles[column], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel('$x_0$', fontsize=8)
            ax.set_ylabel('$x_1$', fontsize=8)
            ax.yaxis.set_label_coords(-.27, .5)
            # Original row annotations collide with the y-axis after reduction.
            for text in list(ax.texts):
                if text.get_text().startswith('Вход '):
                    text.remove()
        fig.canvas.draw()
        for row in range(4):
            boxes = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in fig.axes[row*columns:(row+1)*columns]]
            bounds = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.08)
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=200, bbox_inches=bounds)
            display(Image(data=buffer.getvalue(), width=1150))
        plt.close(fig)
        return None
    width, height = fig.get_size_inches()
    ratio = min(1, 12 / width, 6.8 / height)
    fig.set_size_inches(width * ratio, height * ratio)
    # Preserve the source's manually positioned geometry for multi-stage figures.
    if len(fig.axes) <= 10:
        fig.tight_layout(pad=1.0, h_pad=2.5, w_pad=2.0)
    return fig


def _godoy_wrapper(name):
    def draw(*args, **kwargs):
        return _localized_godoy(name, *args, **kwargs)
    return draw


for _name in ['counter_vs_clock', 'plot_sequences', 'plot_data', 'figure8', 'figure13',
              'canonical_contour', 'hidden_states_contour', 'figure16', 'figure17',
              'figure20', 'figure22', 'figure25']:
    globals()['godoy_' + _name] = _godoy_wrapper(_name)


def plot_sequence_losses(history, title):
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    ax.plot(range(1, len(history['train'])+1), history['train'], label='Обучение', color='#287c73')
    if history['validation']:
        ax.plot(range(1, len(history['validation'])+1), history['validation'],
                label='Валидация', color='#bb4d5b')
    ax.set(xlabel='Эпоха', ylabel='BCE', title=title)
    ax.legend()
    fig.tight_layout()
    return fig


_CHAPTER8_CAPTIONS = {
    'rnn1': 'Single Cell — одна ячейка; Two Identical Cells in Sequence — две одинаковые ячейки последовательно.\nRNN Cell — рекуррентная ячейка; output of this cell — выход этой ячейки;\nto next cell — к следующей ячейке; Tanh — гиперболический тангенс.',
    'rnn2': 'RNN Cell — рекуррентная ячейка. Слева — развёртка по шагам; справа — запись с обратной связью.',
    'rnn_cell_diagram': 'Hidden State — скрытое состояние; Data Points — входные точки.\nTransforming — линейное преобразование; Adding — сложение; Activating — применение активации.\nИндексы 0 и 1 здесь обозначают компоненты вектора, а не шаги времени.',
    'rnn_cell_diagram_seq': 'Hidden State — скрытое состояние; Data Points — входные точки; Transforming — линейное преобразование.\nAdding Transf. Input — добавление преобразованного входа; Activating — применение активации.\nВнутри каждой ячейки индексы 0 и 1 обозначают компоненты состояния.',
    'stacked_rnn': 'Layer — слой; RNN Cell — рекуррентная ячейка. Выход нижнего слоя становится входом верхнего.',
    'bidirect_rnn': 'Forward Layer — прямое направление; Reverse Layer — обратное направление.\nRNN Cell — рекуррентная ячейка; Rev — ячейка обратного направления.',
    'gru_cell': 'Update — гейт обновления; Reset — гейт сброса; Sig — сигмоида; Tanh — гиперболический тангенс.\nЗвёздочка обозначает поэлементное умножение.',
    'lstm_cell': 'Input — входной гейт; Forget — гейт забывания; Output — выходной гейт.\nRNN Cell — рекуррентная ячейка; Sig — сигмоида; Tanh — гиперболический тангенс.\nЗвёздочка обозначает поэлементное умножение.',
    'packed_seq_inddata': 'Sequence — последовательность; Length — длина; Columns — столбцы.\nunsorted_indices — индексы восстановления исходного порядка; batch_sizes — число элементов на шаге.\nЦвета обозначают разные последовательности; # — позиция в упакованном data.',
    'packed_seq_data': 'data — плоская последовательность действительных элементов; # — индекс в data.',
    'conv1_ma': 'Temperatures — температуры; Days — дни; Step 1 — первый шаг.\nОкно содержит пять весов; для скользящего среднего каждый равен 1/5.',
    'conv1d': 'Sequence — последовательность; Filter — фильтр; Element-wise Multiplication — поэлементное умножение.\nFull Output — полный выход; adds up to — сумма; N × C × L — батч × каналы × длина.',
    'conv1d_dilated': 'Sequence — последовательность; Filter — фильтр; Element-wise Multiplication — поэлементное умножение.\nFull Output — полный выход; dilation — расстояние между соседними позициями ядра.',
    'conv1d_edges': 'Sequence — последовательность; Filter — фильтр; Element-wise Multiplication — поэлементное умножение.\nFull Output — полный выход; adds up to — сумма. Здесь показаны веса и числа из книги,\nа не результат текущего обучения; его веса выведены в предыдущей ячейке.',
}


def plot_chapter8_diagram(name):
    """Display the unmodified Godoy PNG with a separate Russian legend."""
    from pathlib import Path

    if name not in _CHAPTER8_CAPTIONS:
        raise ValueError(f'Unknown Chapter 8 diagram: {name}')
    path = Path(__file__).parent / 'assets' / 'godoy08' / f'{name}.png'
    pixels = plt.imread(path)
    height, width = pixels.shape[:2]
    image_width = min(10, 5 * width / height)
    image_height = image_width * height / width
    caption = _CHAPTER8_CAPTIONS[name]
    caption_height = .22 * len(caption.splitlines()) + .25
    figure_width = max(9, image_width)
    fig = plt.figure(figsize=(figure_width, image_height + caption_height),
                     facecolor='white')
    ax = fig.add_axes([0, caption_height / (image_height + caption_height),
                       1, image_height / (image_height + caption_height)])
    ax.imshow(pixels, interpolation='antialiased')
    ax.set_axis_off()
    fig.text(.02, .10 / (image_height + caption_height), caption,
             fontsize=10, color='#202020', va='bottom')
    return fig
