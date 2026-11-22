"""Рисунки дополнительной главы; новая реализация для задачи Годоя."""
import numpy as np
import torch
import matplotlib.pyplot as plt

def array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

# Оформление квадратов: Godoy, plots/chapter8.py::plot_data и
# plots/chapter9.py::sequence_pred (MIT; см. LICENSE-Godoy.txt).
def square(ax, corners, pred=None, title=""):
    corners = array(corners)
    paths = [corners]
    if pred is not None:
        paths.append(np.concatenate([corners[:2], array(pred)], axis=0))
    for j, path in enumerate(paths):
        for i in range(4):
            coords = path[i]
            ax.scatter(*coords, c='k', s=400)
            start = -1 if i == 3 else i
            if not j or i:
                ax.plot(*path[[start, start+1]].T, c='k', lw=2,
                        alpha=.5, linestyle='--' if j else '-')
            ax.text(*(coords - np.array([.04, .04])), str(i+1), c='w', fontsize=12)
    ax.set_title(title, fontsize=14 if pred is None else plt.rcParams["axes.titlesize"])
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$", rotation=0)
    limit = 1.5 if pred is None else 1.45
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])

def plot_data(points, directions, n_cols=4):
    fig, axs = plt.subplots(1, n_cols, figsize=(3*n_cols, 3))
    for i, ax in enumerate(np.atleast_1d(axs)):
        title = "По часовой" if directions[i] else "Против часовой"
        square(ax, points[i], title=f"{title} (y={directions[i]})")
    fig.tight_layout()
    return fig

def sequence_pred(sbs_obj, X, directions=None, n_rows=1, n_cols=4):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    preds = sbs_obj.predict(X[:n_rows*n_cols, :2])
    for i, ax in enumerate(np.atleast_1d(axs).flat):
        title = f"Пример {i}"
        if directions is not None:
            title = "По часовой" if directions[i] else "Против часовой"
        square(ax, X[i], preds[i], title=title)
    fig.tight_layout()
    return fig

def attention_axes(ax, matrix, keys, queries, title):
    # Godoy, chapter9.plot_attention: gray, [0, 1], подписи сверху,
    # контраст текста относительно половины максимального веса.
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap=plt.cm.gray)
    ax.grid(False)
    ax.set_xticks(np.arange(matrix.shape[1]), labels=keys)
    ax.set_yticks(np.arange(matrix.shape[0]), labels=queries)
    ax.set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0]+1)-.5, minor=True)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_title(title, fontsize=14)
    threshold = im.norm(matrix.max()) / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color="black" if im.norm(matrix[i,j]) > threshold else "white")

def plot_matrix(matrix, title, labels=None):
    m = array(matrix)
    labels = labels or [str(i) for i in range(len(m))]
    fig, ax = plt.subplots(figsize=(5, 5))
    attention_axes(ax, m, labels, labels, title)
    ax.set_ylabel("Запрос")
    fig.tight_layout()
    return fig

def plot_attention(model, full_seq, kind="decoder"):
    was_training=model.training
    model.train()
    with torch.no_grad(): model(full_seq.to(next(model.parameters()).device))
    module=model.encoder.layer.self_attn_heads if kind=="encoder" else model.decoder.layer.self_attn_heads
    alphas=array(module.alphas[:,0])
    model.train(was_training)
    labels=["A","B"] if kind=="encoder" else ["B → C","C → D"]
    keys=["A","B"] if kind=="encoder" else ["B","C"]
    fig, axs = plt.subplots(1, len(alphas), figsize=(3*len(alphas), 3))
    for h, ax in enumerate(np.atleast_1d(axs)):
        attention_axes(ax, alphas[h], keys, labels if h == 0 else ["", ""], f"Голова №{h+1}")
    fig.subplots_adjust(wspace=.8, hspace=1.)
    fig.tight_layout()
    return fig

def plot_comparison(runs, full_test):
    fig,axs=plt.subplots(2,len(runs),figsize=(4*len(runs),8))
    for col,(name,sbs) in enumerate(runs.items()):
        pred=sbs.predict(full_test[:2,:2])
        for row in range(2): square(axs[row,col],full_test[row],pred[row],f"{name} · пример {row}")
    fig.tight_layout()
    return fig

def add_arrow(line, position=None, direction='right', size=15, color=None, lw=2, alpha=1.0, text=None, text_offset=(0 , 0)):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
        
#     if end_ind > 1:
#         end_ind = 0
#     if end_ind < 0:
#         end_ind = 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle='--' if alpha < 1 else '-', alpha=alpha),
        size=size,
    )
    if text is not None:
        line.axes.annotate(text, color=color,
            xytext=(xdata[end_ind] + text_offset[0], ydata[end_ind] + text_offset[1]),
            xy=(xdata[end_ind], ydata[end_ind]),
            size=size,
        )
