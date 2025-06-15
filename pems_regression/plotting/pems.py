import contextily
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import osmnx
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pems_regression.utils import dcn


def plot_PEMS(
    G,
    vals,
    vertex_id,
    normalization,
    ax,
    fig,
    cax,
    vmin=None,
    vmax=None,
    filename=None,
    bbox=None,
    nodes_to_label=[],
    node_size=20,
    alpha=0.6,
    edge_linewidth=0.4,
    cmap_name="viridis",
    cut_colormap=False,
    plot_title=None,
    no_background=False,
):
    n, s, e, w = bbox  # bounds of crossroads
    mean, std = normalization
    vals = vals * std + mean
    vertex_id_dict = {vertex_id[i]: i for i in range(len(vertex_id))}

    if vmin is None:
        vmin = np.min(vals)
    if vmax is None:
        if not cut_colormap:
            vmax = np.max(vals)
        else:
            vmax = np.sort(vals)[
                9 * len(vals) // 10
            ]  # variance to high on distant points

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    colors = []
    for i in range(len(G)):
        if vertex_id_dict.get(i) is not None:
            val = vals[vertex_id_dict[i]]
            colors.append(cmap(norm(val)))
        else:
            colors.append((0, 0, 0, 1))  # black

    osmnx.plot_graph(
        G,
        show=False,
        close=False,
        bgcolor="w",
        node_color=colors,
        node_size=0,
        edge_color="black",
        edge_linewidth=edge_linewidth,
        bbox=(w, s, e, n),
        ax=ax,
    )

    if plot_title is not None:
        ax.set_title(plot_title)

    def cmap_alpha_adjusted(x):
        x = float(x)
        if np.isnan(x):
            result = (1, 1, 1, alpha)
        else:
            result = cmap(x)
            result = [*result[:3], alpha]
        return result

    nodes_to_label_set = set(
        nodes_to_label.ravel().tolist()
    )  # nodes_to_label is nodes with data
    for node in G.nodes:
        if node not in nodes_to_label_set:
            x, y = G.nodes[node]["x"], G.nodes[node]["y"]
            if s < y and y < n and w < x and x < e:  # select points at the crossroads
                val = vals[vertex_id_dict[node]]
                ax.scatter(
                    x,
                    y,
                    s=node_size,
                    color=cmap_alpha_adjusted(norm(val)),
                    edgecolors=(0, 0, 0, alpha),
                    linewidths=0.5,
                )

    for node in nodes_to_label_set:
        x, y = G.nodes[node]["x"], G.nodes[node]["y"]
        if s < y and y < n and w < x and x < e:  # select points at the crossroads
            val = vals[vertex_id_dict[node]]
            ax.scatter(
                x,
                y,
                s=node_size,
                color=cmap_alpha_adjusted(norm(val)),
                edgecolors=(0, 0, 0, alpha),
                linewidths=0.5,
            )
            ax.scatter(x, y, s=1 / 20 * node_size, color="black", alpha=alpha)

    # adding realworld map to the background
    if not no_background:
        contextily.add_basemap(
            ax=ax,
            crs="epsg:4326",
            attribution=False,
            zoom_adjust=0,
            interpolation="sinc",
        )

    if filename is not None and cax is None:
        plt.savefig("plots/{}.svg".format(filename), dpi=1000, bbox_inches="tight")

    if cax is not None:
        if cut_colormap:
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm, cmap),
                orientation="vertical",
                extend="max",
                cax=cax,
            )
        else:
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm, cmap), orientation="vertical", cax=cax
            )
        if filename is not None:
            plt.savefig(
                "plots/{}.svg".format(filename),
                dpi=1000,
                transparent=True,
                bbox_inches="tight",
            )


def plot_prediction(
    prediction,
    filename=None,
    *,
    nx_graph,
    xs,
    orig_mean,
    orig_std,
    vmin,
    vmax,
    xs_train,
    alpha_global,
    alpha_local,
):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plot_PEMS(
        nx_graph,
        dcn(prediction),
        dcn(xs),
        (dcn(orig_mean), dcn(orig_std)),
        ax=ax,
        fig=fig,
        cax=cax,
        vmin=vmin,
        vmax=vmax,
        bbox=(37.450, 37.210, -121.80, -122.10),
        nodes_to_label=dcn(xs_train),
        node_size=30,
        alpha=alpha_global,
        edge_linewidth=0.4,
        cmap_name="plasma",
    )

    inner_ax = ax.inset_axes((0, -0.02, 0.5, 0.5))

    plot_PEMS(
        nx_graph,
        dcn(prediction),
        dcn(xs),
        (dcn(orig_mean), dcn(orig_std)),
        ax=inner_ax,
        fig=fig,
        cax=None,
        vmin=vmin,
        vmax=vmax,
        bbox=(37.330741, 37.315718, -121.883005, -121.903327),
        nodes_to_label=dcn(xs_train),
        node_size=60,
        alpha=alpha_local,
        edge_linewidth=0.4,
        cmap_name="plasma",
    )

    inner_ax.patch.set_edgecolor((0, 0, 0, 0.8))
    inner_ax.patch.set_linewidth(2)

    ax.indicate_inset_zoom(inner_ax, edgecolor=(0, 0, 0, 0.8))

    if filename is not None:
        plt.savefig(
            f"plots/{filename}_prediction.svg",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()


def plot_uncertainty(
    stds,
    filename=None,
    *,
    nx_graph,
    xs,
    orig_std,
    vmin,
    vmax,
    xs_train,
    alpha_global,
    alpha_local,
):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plot_PEMS(
        nx_graph,
        dcn(stds),
        dcn(xs),
        (0, dcn(orig_std)),
        ax=ax,
        fig=fig,
        cax=cax,
        vmin=vmin,
        vmax=vmax,
        bbox=(37.450, 37.210, -121.80, -122.10),
        nodes_to_label=dcn(xs_train),
        node_size=30,
        alpha=alpha_global,
        edge_linewidth=0.4,
        cmap_name="plasma",
    )

    inner_ax = ax.inset_axes((0, -0.02, 0.5, 0.5))

    plot_PEMS(
        nx_graph,
        dcn(stds),
        dcn(xs),
        (0, dcn(orig_std)),
        ax=inner_ax,
        fig=fig,
        cax=None,
        vmin=vmin,
        vmax=vmax,
        bbox=(37.330741, 37.315718, -121.883005, -121.903327),
        nodes_to_label=dcn(xs_train),
        node_size=60,
        alpha=alpha_local,
        edge_linewidth=0.4,
        cmap_name="plasma",
    )

    inner_ax.patch.set_edgecolor((0, 0, 0, 0.8))
    inner_ax.patch.set_linewidth(2)

    ax.indicate_inset_zoom(inner_ax, edgecolor=(0, 0, 0, 0.8))

    if filename is not None:
        plt.savefig(
            f"plots/{filename}_uncertainty.svg",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()
