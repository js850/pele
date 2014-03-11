import argparse
from itertools import izip

import numpy as np
import networkx as nx
import pygraphviz
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches

from pele.storage import Database
from pele.landscape import database2graph
from matplotlib.collections import PathCollection

def reduce_graph(graph, nmax=2):
    cc = nx.connected_components(graph)
    
    # remove all clusters with less than nmax minima
    cc = filter(lambda c: len(c) >= nmax, cc)
    
    nodes = [n for nlist in cc for n in nlist]
    return nx.subgraph(graph, nodes)

def make_graph(db):
    graph = nx.Graph()
    for ts in db.transition_states():
        graph.add_edge(ts.minimum1._id, ts.minimum2._id)
    return graph

def color_minima_by_energy(db, graph):
    minima = db.minima()
    emin = minima[0].energy
    emax = minima[-1].energy
    cmap = cm.ScalarMappable(cmap="hot")
    cmap.set_clim(vmin=emin, vmax=emax)
    print cmap.get_clim()
    print mpl.colors.rgb2hex(cmap.to_rgba(-42))
    for m in minima:
        try:
            c = mpl.colors.rgb2hex(cmap.to_rgba(m.energy))
            print c, m.energy
            graph.node[m._id]["color"] = mpl.colors.rgb2hex(cmap.to_rgba(m.energy))
        except KeyError:
            pass
    

def prepare_graph(db):
    print "loading graph"
    graph = make_graph(db)
    for m in db.minima():
        try:
            graph.node[m]["energy"] = m.energy
        except KeyError:
            pass
    color_minima_by_energy(db, graph)
    graph = reduce_graph(graph)
    
#     m2i = dict(( (m, m._id) for m in graph.nodes_iter() ))
#     graph = nx.relabel_nodes(graph, m2i)
    return graph

def plot_nx(graph):
    nx.draw_graphviz(graph, prog="fdp")
    import matplotlib.pyplot as plt
    plt.show()


def pos2path(pos):
    """string to path"""
    spos = pos.split()
    if len(spos) != 4:
        print "len(spos)", len(spos)
        print pos
        print "error, can't deal with this type of position yet"
    
    points = []
    codes = []
    for point in spos:
        xy = map(float, point.split(","))
        print xy
        points.append(xy)
        if len(codes) == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.CURVE4)
    
    
    path = Path(points, codes)
    return path
    
def get_edge_collection(agraph):
    mpl.collections.PathCollection
    paths = []
    for edge in agraph.edges_iter():
        pos = edge.attr["pos"]
#        print pos
        path = pos2path(pos)
        paths.append(path)
#        patch = patches.PathPatch(path, facecolor='none', lw=2, linestyle="solid")
#        ax.add_patch(patch)
        
    
    path_collection = PathCollection(paths, facecolor="none")
    return path_collection

def get_node_collection(ax, agraph, scale=70.):
    ellipses = []
    widths = []
    heights = []
    offsets = []
    for node in agraph:
        pos = node.attr["pos"]
        xy = map(float, pos.split(","))
        width = float(node.attr["width"])*scale
        height = float(node.attr["height"])*scale
        color = node.attr["color"]
        print xy, width, height
        ellipse = patches.Ellipse(xy, width, height, facecolor='none', edgecolor=color)
        ellipses.append(ellipse)
        widths.append(width)
        heights.append(height)
        offsets.append(xy)
        
#        ax.add_patch(ellipse)
    
    rot = [0. for h in heights]
    offsets = np.array(offsets)
    print offsets
    
#    ellipse_collection = mpl.collections.EllipseCollection(widths, heights, rot, 
#                                                           offsets=offsets,
##                                                           facecolor="none", 
#                                                           units="x",
#                                                           offset_position="data"
#                                                           )
    patch_collection = mpl.collections.PatchCollection(ellipses, match_original=True)
    return patch_collection, offsets

def get_agraph_bounding_box(agraph):
    """there is a bug which means you can't get it the normal way
    
    https://groups.google.com/forum/#!topic/pygraphviz-discuss/QYXumyw3E-g
    """
    return pygraphviz.graphviz.agget(agraph.handle,'bb')

def plot_edges(agraph):
    # get the bounding box
    bb = get_agraph_bounding_box(agraph)
    xmin, ymin, xmax, ymax = map(float, bb.split(","))
    
    # determine the figure size
    scale = np.sqrt(6*8 / (xmax*ymax))
    fxmax = xmax * scale 
    fymax = ymax * scale
    print "xmax, ymax",  fxmax, fymax, xmax, ymax, scale

    fig = plt.figure(figsize=(fxmax, fymax))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # add the edges
    path_collection = get_edge_collection(agraph)
    ax.add_collection(path_collection)
    
    # add the nodes
    point_collection, points = get_node_collection(ax, agraph)
    ax.add_collection(point_collection)
    
    # label the nodes
    fontsize = 8
    for (x, y), node in izip(points, agraph):
        label = str(node)
        ax.text(x, y, label, horizontalalignment='center',
                verticalalignment='center', size=fontsize)
    
#    lims = agraph.graph_attr["bb"]
    
    fig.set_size_inches(xmax, ymax)
    
#    fig.draw()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="make a graphviz dot file from the database")

    parser.add_argument("--db", type=str, default=None, help="Database file name")
    parser.add_argument("-o", type=str, default="dbgraph.dot", help="output name")
    parser.add_argument("--PS", action="store_true", help="use a pathsample database")
    args = parser.parse_args()
    
    if args.db is None and not args.PS:
        parser.print_help()
        print "you must give either a database or specify --PS"
        exit()
    
    db = Database(args.db, createdb=False)
    
    print "preparing graph"
    graph = prepare_graph(db)
    print "writing dot file"
    nx.write_dot(graph, args.o)
    
#     plot_nx(graph)
    agraph = nx.to_agraph(graph)
    agraph.graph_attr["splines"] = True
    agraph.graph_attr["overlap"] = False
    agraph.layout(prog="sfdp")
    agraph.write(open("test.dot", "w"))
    plot_edges(agraph)

    graph.graph["splines"] = True
    graph.graph["overlap"] = False
    nx.view_pygraphviz(graph, prog="sfdp")

if __name__ == "__main__":
    main()