import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import shapely as sl
import numpy as np


def compute_centroid(vertices):
    num_vertices = len(vertices)
    if num_vertices == 0:
        return None
    
    sum_x = sum(vertex[0] for vertex in vertices)
    sum_y = sum(vertex[1] for vertex in vertices)
    
    centroid_x = sum_x / num_vertices
    centroid_y = sum_y / num_vertices
    
    return centroid_x, centroid_y


def rotate_line(start, end, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.deg2rad(angle_degrees)
    # Translate end point by the negative of start point
    translated_end = (end[0] - start[0], end[1] - start[1])
    # Perform rotation
    rotated_end = (translated_end[0]*np.cos(angle_radians) - translated_end[1]*np.sin(angle_radians),
                   translated_end[0]*np.sin(angle_radians) + translated_end[1]*np.cos(angle_radians))
    # Translate end point back
    rotated_end = (rotated_end[0] + start[0], rotated_end[1] + start[1])
    
    return rotated_end


def attractive_force(current_position, final_position):
    diffx, diffy = (final_position[0] - current_position[0], final_position[1] - current_position[1])
    d_Fa = (diffx, diffy) / np.sqrt(diffx**2 + diffy**2)
    
    k_att = 1.5
    alpha = 0.2
    # Compute the angle between the current direction and the line to the goal
    angle = np.arctan2(diffy, diffx)  # Angle with respect to the x-axis
    
    # Adjust the magnitude of the force based on the angle (conic shape)
    magnitude = k_att *np.sqrt(d_Fa[0]**2 + d_Fa[1]**2) * np.cos(alpha * (angle - np.pi / 2))**2
    
    # Compute the force components
    Fa = d_Fa
    Fa[0] = magnitude * np.cos(angle)
    Fa[1] = magnitude * np.sin(angle)
    
    return Fa


def repulsive_force(current_position, obstacle_pts, ro=.3):
    '''Repulsive force function for use in potential navigation function:
    INPUT: 
        - current position q as an array of length 2
        - An (N,2) array of all points along all obstacles
        - Step Size as a scalar
        - ro: distance in x,y at which to start considering repulsive force. Enter values between 0 and 1. The threshold.
    OUTPUT: 
        - An array of length 2 representing the repulsive force to be added to the total force in x and y directions
        '''
    
    ro = ro*np.array([1,1]) # defining the dist threshold in x and y where we need to pay attention!
    r = np.zeros((len(obstacle_pts),2))
    
    for i in range(len(obstacle_pts)):
        obstacle_position = obstacle_pts[i]
        diffx, diffy = (obstacle_position[0] - current_position[0], obstacle_position[1] - current_position[1])
        r[i] = (diffx, diffy) / np.sqrt(diffx**2 + diffy**2)

    rq = np.array([np.min(r[:,0]), np.min(r[:,1])])

    # Return the repulse force, Fr
    if (rq[0] < ro[0]) or (rq[1] < ro[1]):
        # normalizing
        vect = (1/2)*((1/rq - 1/ro)**2)
        return vect / np.linalg.norm(vect)
    else:
        return np.array([0.,0.])


def potential_field(obstacle_pts,lines,position=(2,1),qf=(9,4),step=0.05,ro=0.5):
    '''
    Potential field implementation with a predefined graph.
    Flexible start and end position, obstacle locations are locked. Need to enter the points that represent the obstacles,
    and the lines that define the boundaries.
    '''
    # Define intital parameters
    ttl_dist = 0
    
    # Plot grid
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(range(11))
    ax.set_yticks(range(7))
    ax.grid(True)

    for i in range(500):
        
        position_new = step*(1.5*attractive_force(position, qf) + repulsive_force(position, obstacle_pts, ro=.5)) + position
        # position_new = attractive_force(position, qf)*step + position (update of position using only attractive force.)
        
        a = sl.LineString([position, position_new])
        
        # For each obstacle line segment, if we will intersect it, then rotate our line
        for line in lines:
            b = sl.LineString(line)
            if a.intersects(b):
                while(True):
                    endpt_rot = rotate_line(position, position_new, 40) # rotate the line by 30 degrees
                    a = sl.LineString([position, endpt_rot]) # update a as the new line and check again
                    position_new = endpt_rot # update the position_new
                    if not (a.intersects(b)):
                        break  
        
        # Add total distance traveled
        segment_dist = np.sqrt((position[0] - position_new[0])**2 + (position[1] - position_new[1])**2)
        ttl_dist += segment_dist
        
        # Plotting the new segment
        x = [position[0],position_new[0]]
        y = [position[1],position_new[1]]
        ax.plot(x,y,color='blue')
        
        position = position_new
        
        # Set vertices of the obstacles
    CB1 = [(3,3),(3,4),(5,4),(5,3)]  # Rectangle
    CB2 = [(7,4),(7,2),(8,2)]        # Triangle

    # Plot rectangle
    rect = Rectangle((CB1[0][0], CB1[0][1]), CB1[2][0]-CB1[0][0], CB1[2][1]-CB1[0][1], edgecolor='red', facecolor='none',linewidth=3)
    ax.add_patch(rect)
    # Plot triangle
    triangle = Polygon(CB2, closed=True, edgecolor='blue', facecolor='none',linewidth=3)
    ax.add_patch(triangle)

    # # Plot start and end position
    qo = (2,1)
    qf = (9,4)
    ax.scatter(2,1,c='b')
    ax.scatter(9,4,c='r')

    # Show plot
    plt.title('2D Potential Field Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    return ttl_dist

def exact_cell_decomposition(vertices, lines):
    '''
    decomposes a 2D space into regions using a line sweep algorithm, then finds centroids of those regions.
    Returns a plot of the decomposed region and the list of centroids of the regions.
    '''
    # Set vertices of the obstacles
    CB1 = [(3,3),(3,4),(5,4),(5,3)]  # Rectangle
    CB2 = [(7,4),(7,2),(8,2)]        # Triangle
    # Plot grid
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(range(11))
    ax.set_yticks(range(7))
    ax.grid(True)
    # Plot rectangle
    rect = Rectangle((CB1[0][0], CB1[0][1]), CB1[2][0]-CB1[0][0], CB1[2][1]-CB1[0][1], edgecolor='red', facecolor='none',linewidth=3)
    ax.add_patch(rect)
    # Plot triangle
    triangle = Polygon(CB2, closed=True, edgecolor='blue', facecolor='none',linewidth=3)
    ax.add_patch(triangle)
    # Plot start and end position
    qo = (2,1)
    qf = (9,4)
    ax.scatter(2,1,c='b')
    ax.scatter(9,4,c='r')

    regions = [] # for saving the lines added

    for i in range(10):
        
        a = sl.LineString([(i+1,0), (i+1,6)]) # vertical line from bottom to top
        # For each intersection with an obstacle vertex, we draw a vertical line from top to bottom
        for vertex in vertices:
            if sl.intersects(a,sl.Point(vertex)):
                ax.plot((i+1,i+1),(0,6),color='green')
                regions.append(a) # appends a tuple of tuple (a line start and end point)
                break
        
    # Segment into distinct polygon regions
    polygon = []
    centroids = []
    dist_prev = 0
    for i in range(len(regions)):
        
        dist_next = sl.distance(regions[i],sl.Point(0,0))
        poly = ((dist_prev,0),(dist_next,0),(dist_prev,6),(dist_next,6)) # defines the vertices of the new region
        
        # check if the centroid of the poly intersects a line of an obstacle
        for line in lines:
            a = sl.LineString(line)
            # If the centroid intersects a line, split into two regions
            if sl.intersects(a,sl.Point(compute_centroid(poly)[0], compute_centroid(poly)[1])): # if centroid intersects an obstacle line, decompose
                # we split into two regions, above the obstacle, and below
                poly1 = ((dist_prev,0),(dist_next,0),(dist_prev,3),(dist_next,3))
                poly2 = ((dist_prev,4),(dist_next,4),(dist_prev,6),(dist_next,6))
                polygon.append(poly1)
                polygon.append(poly2)
                centroids.append(compute_centroid(poly1))
                centroids.append(compute_centroid(poly2))
                ax.scatter(compute_centroid(poly1)[0],compute_centroid(poly1)[1],color='orange')
                ax.scatter(compute_centroid(poly2)[0],compute_centroid(poly2)[1],color='orange')
                # FIXME: Make the split based on the vertices of the objects
        
        # if centroid doesn't intersect an obstacle line, plot it
        for line in lines:
            a = sl.LineString(line)
            if sl.intersects(a,sl.Point(compute_centroid(poly)[0], compute_centroid(poly)[1])): 
                break
            else:
                if sl.Point(compute_centroid(poly)[0], compute_centroid(poly)[1]) == sl.Point(7.5,3): # special case
                    break    
                polygon.append(poly)
                centroids.append(compute_centroid(poly))
                ax.scatter(compute_centroid(poly)[0],compute_centroid(poly)[1],color='orange')
                    
        dist_prev = dist_next
        
    # Show plot
    plt.title('2D Continous Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    return centroids        
         