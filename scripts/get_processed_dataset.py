"""
This script is used to preprocess the raw PEMS data into a graph along with
continuous node labels on a subset of its nodes. The graph is a real-world graph
of roads in California, and the node labels are the traffic speed data from the
California PEMS.

Written originally by Iskander Azangulov, adapted by Viacheslav Borovitskiy.
"""

import argparse
import os
import pickle
import tempfile
import zipfile

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        print("Failed to cut the edge")
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[: i + 1]),
                LineString(coords[i:]),
            ]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:]),
            ]


def load_PEMS(raw_data_path: str):  # noqa: C901
    # Use a temp directory for extraction
    tmpdir = tempfile.mkdtemp()

    # Unzip data into tmpdir
    with zipfile.ZipFile(raw_data_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir)

    # Data reading
    adj_path = os.path.join(tmpdir, "PEMS", "adj_mx_bay.pkl")
    signals_path = os.path.join(tmpdir, "PEMS", "pems-bay.h5")
    coords_path = os.path.join(tmpdir, "PEMS", "graph_sensor_locations_bay.csv")

    with open(adj_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, _ = pickle.load(f, encoding="latin1")
    all_signals = pd.read_hdf(signals_path)
    coords = pd.read_csv(coords_path, header=None)

    # Loading real world graph of roads
    north, south, east, west = 37.450, 37.210, -121.80, -122.10
    graph_path = os.path.join(tmpdir, "PEMS", "bay_graph.pkl")
    if not os.path.isfile(graph_path):
        cf = '["highway"~"motorway|motorway_link"]'  # Road filter, we don't use small ones.
        G = osmnx.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            simplify=True,
            custom_filter=cf,
        )
        with open(
            graph_path, "wb"
        ) as f:  # frequent loading of maps leads to a temporary ban
            pickle.dump(G, f)
    else:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

    G = osmnx.convert.to_undirected(G)  # Matern GP supports only undirected graphs.

    # Graph cleaning up
    for _ in range(2):
        out_degree = G.degree
        to_remove = [node for node in G.nodes if out_degree[node] == 1]
        G.remove_nodes_from(to_remove)
    G = nx.convert_node_labels_to_integers(G)
    G.remove_nodes_from([372, 286])
    G = nx.convert_node_labels_to_integers(G)

    num_points = len(sensor_ids)

    np_coords = np.zeros((num_points, 2))  # Vector of sensors coordinates.
    for i in range(num_points):
        sensor_id, x, y = coords.iloc[i]
        ind = sensor_id_to_ind[str(int(sensor_id))]
        np_coords[ind][0], np_coords[ind][1] = x, y
    coords = np_coords

    sensor_ind_to_node = {}
    print(G.graph["crs"])
    # Inserting sensors into a graph. During insertion, the edge containing the sensor is split
    G = osmnx.project_graph(G, to_crs=3857)
    for point_id in range(num_points):
        sensor_ind_to_node[point_id] = len(G)  # adding new vertex at the end
        sensor_point = gpd.GeoSeries(
            Point(coords[point_id, 1], coords[point_id, 0]), crs=4326
        ).to_crs(3857)[0]

        u, v, key = osmnx.distance.nearest_edges(G, sensor_point.x, sensor_point.y)
        edge = G.edges[(u, v, key)]
        geom = edge["geometry"]
        u_, v_ = u, v
        if (G.nodes[u]["x"] - geom.coords[0][0]) ** 2 + (
            G.nodes[u]["y"] - geom.coords[0][1]
        ) ** 2 > 1e-6:
            u, v = v, u
        G.remove_edge(u_, v_, key)

        edge_1_geom, edge_2_geom = cut(geom, geom.project(sensor_point))
        l_ratio = geom.project(sensor_point, normalized=True)
        l_1, l_2 = l_ratio * edge["length"], (1 - l_ratio) * edge["length"]
        new_vertex = nearest_points(geom, sensor_point)[0]
        G.add_node(len(G), x=new_vertex.x, y=new_vertex.y)
        G.add_edge(u, len(G) - 1, length=l_1, geometry=edge_1_geom)
        G.add_edge(len(G) - 1, v, length=l_2, geometry=edge_2_geom)
    G = osmnx.project_graph(G, to_crs=4326)
    G = osmnx.convert.to_undirected(G)

    new_G_labels = {node: id_ for id_, node in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, new_G_labels)
    sensor_ind_to_node = {
        sensor_ind: new_G_labels[node]
        for (sensor_ind, node) in sensor_ind_to_node.items()
    }

    # Weights are inversely proportional to the length of the road
    lengths = nx.get_edge_attributes(G, "length")
    lengths_list = [length for length in lengths.values()]
    mean_length = np.mean(lengths_list)
    weights = {}
    for edge, length in lengths.items():
        weights[edge] = mean_length / length
    nx.set_edge_attributes(G, values=weights, name="weight")

    # sensor_ind - sensor id in California database, sensor_id - local numeration (from 1 to num of sensors)
    sensor_id_to_node = {}
    for sensor_ind, node in sensor_ind_to_node.items():
        sensor_id = sensor_ids[sensor_ind]
        sensor_id_to_node[sensor_id] = node

    # Selecting signals at some moment
    signals = all_signals[
        (all_signals.index.weekday == 0)
        & (all_signals.index.hour == 17)
        & (all_signals.index.minute == 30)
    ]

    # Dataset creation
    x, y = [], []
    for i in range(len(signals)):
        for sensor_id in sensor_ids:
            if sensor_id_to_node.get(sensor_id) is not None:
                node = sensor_id_to_node[sensor_id]
                signal = signals.iloc[i][int(sensor_id)]
                x.append([i, node])
                y.append([signal])

    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x, y = x[:num_points, 1:], y[:num_points]

    return G, (x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the raw PEMS data into a graph along with continuous node labels on a subset of its nodes"
    )
    parser.add_argument(
        "raw_data_path", type=str, help="Path to the zip archive with the raw PEMS data"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the processed data as a pickle file",
    )
    args = parser.parse_args()
    print("Using raw data from", args.raw_data_path)
    G, data = load_PEMS(args.raw_data_path)
    output_path = os.path.join(args.output_dir, "processed_pems_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((G, data), f)
    print(f"Processed data saved to {output_path}")
