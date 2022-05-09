import numpy as np

def get_heatmap(self, col):
    x = self.unique_vals[col]
    t = self.unique_vals[0]

    x = np.append(x, round(x[-1]))
    t = np.append(t, round(t[-1]))
    assert (len(x) == len(t))
    N = len(x)

    X         = np.zeros((N**2, N**2))
    X[:, col] = np.repeat(x, N)
    X[:, 0]   = np.tile(t,   N)

    preds     = self._get_F_Xt(X, col)
    preds     = preds.reshape((N, N))
    preds     = preds[1:, 1:]

    N = len(np.unique(preds))

    font_size = 36
    #plt.subplots_adjust(left=0.14, bottom=0.14, right=0.98, top=0.98)
    plt.xticks(fontsize= font_size)
    plt.yticks(fontsize= font_size)
    y_ticks = range(0, round(t[-1])+1)
    x_min = round (x.min())
    x_max = round (x.max())
    x_n_ticks = 5
    x_ticks = [x_min+(x_max-x_min)*i/x_n_ticks for i in range(x_n_ticks+1)]
    cbar_min = -1#round (preds.min(), 1)
    cbar_max = +1#round (preds.max(), 1)
    cbar_n_ticks = 5
    cbar_ticks = [cbar_min+(cbar_max-cbar_min)*i/cbar_n_ticks for i in range(cbar_n_ticks+1)]
    fig, ax = plt.subplots(figsize=(19, 12), dpi=100)
    colormesh = ax.pcolormesh(t, x, preds, cmap='plasma', vmin=cbar_min, vmax=cbar_max, linewidths=0.1)
    ax.set_xticks(y_ticks)
    ax.set_xticklabels(y_ticks, fontsize=font_size)
    ax.set_xlabel("time", fontsize=font_size)
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(x_ticks, fontsize=font_size)

    title = r"$F_{"+str(self.num_iters).encode('unicode_escape').decode()+"}(t, x_{"+str(col).encode('unicode_escape').decode()+"})="+r"-\nu\sum"+r"_{m=0}^{"+str(self.num_iters-1).encode('unicode_escape').decode()+r"}g_m(t,x_{"+str(col).encode('unicode_escape').decode()+r"})$"
    ax.set_title(title, fontsize = font_size-4)

    ax.set_ylabel((r"$X_{"+str(col).encode('unicode_escape').decode()+r"}$"), fontsize=font_size)
    cbar = fig.colorbar(colormesh, ax=ax)
    cbar.set_ticks(cbar_ticks, cbar_ticks)
    cbar.ax.tick_params(labelsize=font_size)
    return fig, ax