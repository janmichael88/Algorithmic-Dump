#code written from psuedo code
def dist_func(x,y):
    return(np.sqrt(x**2 + y**2))

#DB, points to search
#G is point to dpeth search from
#eps is neighboor radiaus
def range_query(DB, dist_func, Q, eps):
    Neighbors = []
    for point in DB:
        if dist_func(Q,point) <= eps:
            Neighbors.append(point)
        else:
            pass
    return(Neighbors)

def find_label(p,DB):
    #create DB with column of empty responses
    num_observations = DB.shape[0]
    #create zeros vector
    DB = np.vstack([DB])
def DBSCAN(DB, distfunc,eps,minPts):
    #set cluster counter
    C = 0
    for point in DB:
        if point not in clusters:
            #find neighbors
            Neighbors = range_query(DB,dist_func,point,eps)
            if len(Neighbors) < minPts::
                #classifiy as noise
        else: 
            continue
        




#N is the set of nodes, list 
#e is the set of edges between nodes,dict key is node pair, value is distance
#e node pairs are not repeated in e
ReachedNodes = list(N)[0]
UnreachedNodes = [node for node in list(N)[1:-2]]
SpanningTree = []

def find_min_cost_next_node(a,dict_distance):
    #finds the nearest node from a
    pairs = [x for x,y in dict_distance.items()]
    edges = [y for x,y in dict_distance.items()]
    if a in [node for node[0] in pairs]:
        #find which pairs have a
        index_a = [node for node[0] in pairs] == a
        nodes_a = pairs[index_a]
        #find distances where nodes_a in edges
        distances_from_a = edges[index_a]
        min_distance = min(distances_from_a)
        #return nodes having min distance
        pair_min_distance = pair[edges == min_distance]
        return(pair_min_distance[pair_min_distance != a],pair_min_distance)

while (len(UnreachedNodes) != 0):
    #traverse unreached nodes
    for i in range(0,len(UnreachedNodes)):
        #find neartest node
        nearest_node,edge = find_min_cost_next_node(UnreachedNodes[i],e)
        #add to Reached Nodes
        ReachedNodes.append(nearest_node)
        #remoive from unreached nodes
        UnreachedNodes.remove(nearest_node)
        #add edge to min spanning tree
        SpanningTree.add(edge)



