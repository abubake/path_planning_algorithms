import heapq

'''
This file defines a map class for use in path planning algorithms, as well as a Dijkstra implementation, with which a shortest path can be found if a map is defined.
'''

class BakerGraph:
    '''
    Defines two dictionaries:
        - Vertices: 'A': (x,y)
        - Vertice_edges: 'A': [('B', 4), ('C', 2), ('B', 4), ('C', 2)]
        
    These store all the needed information for our graph. Note edges store the connecting node AND the distance / weight to it
    '''
    def __init__(self):
        self.vertices = {}
        self.vertice_edges = {} # takes the historic role of dictionary of edges for each node

    def add_vertex(self, vertex, x, y):
        self.vertices[vertex] = (x, y)
        
    def add_vertex_edge(self, vertex):
        self.vertice_edges[vertex] = []

    def add_edge(self, vertex1, vertex2, weight):
        if vertex1 in self.vertices and vertex2 in self.vertices:
            self.vertice_edges[vertex1].append((vertex2, weight))
            self.vertice_edges[vertex2].append((vertex1, weight))
            

class lookAtThisGraph:
    '''
    Depreciated
    '''
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, vertex):
        self.vertices[vertex] = []

    def add_edge(self, vertex1, vertex2, weight):
        self.vertices[vertex1].append((vertex2, weight))
        self.vertices[vertex2].append((vertex1,weight))
        
        
def dijkstra(map_object, start_vertex='A', destination_vertex='F'):
    ''' Deprecated
    Dijsktra: it just finds the shortest path in a map, given the map is BakersGraph or lookAtThisGraph.
    
    Note: works with add_vertex and add_edge functions of the graph.
    '''
    # Assign all distances to infinity and build a dictionary of distances for each node
    distances = {vertex: float('infinity') for vertex in map_object.vertices}
    distances[start_vertex] = 0 # assigning distance from start vertex to 0
    
    # creates a que, which is a list of tuples containing distance and vertex
    priority_queue = [(0, start_vertex)]
    
    while True:
        # Return the smallest element from the que
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        if current_vertex == destination_vertex:
            break # If we have reached the destination
        
        if current_distance > distances[current_vertex]:
            continue # do nothing and start the loop over again 
        
        # For the current vertex we are assessing, iterate through the tuples of (neighbor_node, distance)
        for neighbor, weight in map_object.vertice_edges[current_vertex]:
            dist = current_distance + weight # adds distance of new node the currently shortest path we pulled off the que
            
            # update distance dictionary for the given node
            if dist < distances[neighbor]:
                distances[neighbor] = dist
                heapq.heappush(priority_queue, (dist, neighbor))
        
    return distances[destination_vertex], distances


def dijkstra_with_path(map_object, start_vertex='A', destination_vertex='F'):
    '''
    Finds shortest path in a graph of vertices connected with edges.
    
    Inputs: 
        - a map object, basically has to be a "BakersGraph"
        - a string as a designator for the starting vertex
        - a string as a designator for the desination vertex
    Output:
        - The scalar distance from start to end vertex
        - A list of the string names of the vertex's visited from start to end
    '''
    
    # Assign all distances to infinity and build a dictionary of distances for each node
    distances = {vertex: float('infinity') for vertex in map_object.vertices}
    distances[start_vertex] = 0  # assigning distance from start vertex to 0
    
    # creates a queue, which is a list of tuples containing distance and vertex
    priority_queue = [(0, start_vertex)]
    visited = set()  # Set to keep track of visited nodes
    predecessors = {}  # Dictionary to store predecessors for each vertex
    
    while priority_queue:
        # Return the smallest element from the queue
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        if current_vertex == destination_vertex:
            break  # If we have reached the destination
        
        if current_vertex in visited:
            continue  # Skip processing if the vertex has already been visited
        
        # Mark current vertex as visited
        visited.add(current_vertex)
        
        # For the current vertex we are assessing, iterate through the tuples of (neighbor_node, distance)
        for neighbor, weight in map_object.vertice_edges[current_vertex]:
            dist = current_distance + weight  # adds distance of new node to the currently shortest path we pulled off the queue
            
            # Update distance dictionary for the given node
            if dist < distances[neighbor]:
                distances[neighbor] = dist
                predecessors[neighbor] = current_vertex
                heapq.heappush(priority_queue, (dist, neighbor))
        
    # Reconstruct path from start to end
    path = []
    current_vertex = destination_vertex
    while current_vertex != start_vertex:
        path.insert(0, current_vertex)
        current_vertex = predecessors[current_vertex]
    path.insert(0, start_vertex)
    
    return distances[destination_vertex], path


############################ Example usage #################################################################
g = BakerGraph()

# Define the boundaries of the rectangular region
x_min, x_max = 0, 10
y_min, y_max = 0, 6
num_pts = 6

from scipy.stats import uniform
import numpy as np

# Generate the points
x_pts = uniform.rvs(loc=x_min, scale=x_max-x_min, size=num_pts)
y_pts = uniform.rvs(loc=y_min, scale=y_max-y_min, size=num_pts)
uniform_pts = np.column_stack((x_pts, y_pts))

# Add vertices
vertices = ['A', 'B', 'C', 'D', 'E', 'F']
for i, vertex in enumerate(vertices):
    g.add_vertex(vertex,x_pts[i],y_pts[i])
    g.add_vertex_edge(vertex)
    
# Add edges
g.add_edge('A','B', 4)
g.add_edge('A','C', 2)
g.add_edge('B', 'A', 4)
g.add_edge('B','C', 1)
g.add_edge('B', 'D', 5)
g.add_edge('C','A', 2)
g.add_edge('C','B', 1)
g.add_edge('C','D', 8)
g.add_edge('C','E', 10)
g.add_edge('D','B', 5)
g.add_edge('D','C', 8)
g.add_edge('D','E',2)
g.add_edge('D','F',6)
g.add_edge('E','C', 10)
g.add_edge('E','D', 2)
g.add_edge('E','F',5)
g.add_edge('F','D',6)
g.add_edge('F','E',5)

# Find the shortest path from A to F
start_vertex = 'A'
destination_vertex = 'F'
shortest_path_length = dijkstra_with_path(g, 'A', 'F')

print(f"Shortest Path from {start_vertex} to {destination_vertex}: {shortest_path_length}")
    

