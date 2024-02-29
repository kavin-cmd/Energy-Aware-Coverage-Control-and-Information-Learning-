import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import multivariate_normal
##########################################################################
# Implemented by Aiman Munir, Ph.D. Candidate - School of Computing, University of Georgia
##########################################################################

# distance Calculation
def dist(x, y, pos):
    return math.sqrt(((pos[0]-x)**2) + ((pos[1]-y)**2))
 
# partitionFinder
def partitionFinder(ax, robotsPositions, envSize_X, envSize_Y, resolution, densityArray, distribution):
    hull_figHandles = []
    num_robots = robotsPositions.shape[0]
    distArray = np.zeros(num_robots)
    colorList = ["red","green","blue","black","grey","orange"]
    locations = [[] for _ in range(num_robots)]
    robotDensity = [[] for _ in range(num_robots)]
    locationsIdx = [[] for _ in range(num_robots)]
    text_handles = []

    
    # for partition display 
    # partition display require finer resolution
    resolution_display = 0.02
    x_global_values = np.arange(envSize_X[0], envSize_X[1] + resolution_display, resolution_display) 
    y_global_values = np.arange(envSize_Y[0], envSize_Y[1] + resolution_display, resolution_display)
    
    
    for i, x_pos in enumerate(x_global_values):
        for j, y_pos in enumerate(y_global_values):
            for r in range(num_robots):    
                distanceSq = (robotsPositions[r, 0] - x_pos) ** 2 + (robotsPositions[r, 1] - y_pos) ** 2
                #distArray[r] = abs(math.sqrt(distanceSq))
                distArray[r] = abs(math.sqrt(distanceSq))
            minValue = np.min(distArray)
            minIndices = np.where(distArray == minValue)[0]
            for r in minIndices:
                locations[r].append([x_pos, y_pos])
            #minIndex = np.argmin(distArray)
            #locations[minIndex].append([x_pos, y_pos])
    
    # There is no-builtin library in python that supports good visulaization for voronoi
    # Therefore, using convex hull to draw the boundary of each partition
    # It requires deleting previous boundaries and plotting new ones at every iteration
    # Need this object handles so we can remove previous boundaries before calling the partitionFinder function from main file
    for r in range(num_robots):
        robotsLocation = np.array(locations[r])
        if len(robotsLocation)!=0:
            hull = ConvexHull(robotsLocation)
            # Get the vertices of the convex hull
            boundary_points = robotsLocation[hull.vertices]
            # Extract x and y coordinates
            x, y = boundary_points[:, 0], boundary_points[:, 1]
            hullHandle, =  ( ax.plot(x, y, marker='None', linestyle='-', color="black", markersize=6, linewidth =4))
            hull_figHandles.append(hullHandle)
    
    
    # Voronoi Partitioning
    x_global_values = np.arange(envSize_X[0], envSize_X[1] + resolution, resolution) 
    y_global_values = np.arange(envSize_Y[0], envSize_Y[1] + resolution, resolution)
    
    c_v = np.zeros((num_robots,2))
    C_x = np.zeros(num_robots)
    C_y = np.zeros(num_robots)
    Mass = np.zeros(num_robots)
    locationalCost = 0
    locations_new = [[] for _ in range(num_robots)]

    
    for i,ix in enumerate(x_global_values):
        for j,iy in enumerate(y_global_values):
            importance_value = densityArray[i,j] # from the surrogate * resolution * resolution #1
            phi_value = distribution[i,j] # for Locational cost
            
            distArray = np.zeros(num_robots)
            for robots in range(num_robots):
                #distanceSq = (robotsPositions[robots, 0] - ix) ** 2 + (robotsPositions[robots, 1] - iy) ** 2
                #distArray[robots] = abs(math.sqrt(distanceSq))
                distArray[robots] = (robotsPositions[robots, 0] - ix) ** 2 + (robotsPositions[robots, 1] - iy) ** 2 # Squared of the distance
            min_index = np.argmin(distArray)
            c_v[min_index][0] += ix * importance_value
            c_v[min_index][1] += iy * importance_value
            Mass[min_index] +=  importance_value 
            locationalCost += phi_value * distArray[min_index]  # locational cost considers the importance value from the ground truth distribution (not the surrogate)
            locations_new[min_index].append([ix,iy])
            locationsIdx[min_index].append([i,j])
      
    for robots in range(num_robots):       
       if not Mass[robots] == 0:
          C_x[robots] = c_v[robots][0] / Mass[robots]
          C_y[robots] = c_v[robots][1] / Mass[robots]  
    
    '''
    
    # for centroid calculation
    # centroid calculation can be done using lower resolution
    # Eq 1 from paper [1]
    # LocationalCost Equation: H(x, phi) = sum(h_i(x, phi)) = sum(integral_{V_i(x)} ||q - x_i||^2 * phi(q) dq) for i from 1 to N
    # Eq 2 from paper [1]
    # Cenroid Equation: c_i(x) = integral_{V_i(x)} q * phi(q) dq / integral_{V_i(x)} phi(q) dq
    x_global_values = np.arange(envSize_X[0], envSize_X[1] + resolution, resolution) 
    y_global_values = np.arange(envSize_Y[0], envSize_Y[1] + resolution, resolution)
    locations_new = [[] for _ in range(robotsPositions.shape[0])]
    for i, x_pos in enumerate(x_global_values):
        for j, y_pos in enumerate(y_global_values):
            for r in range(robotsPositions.shape[0]):    
                distanceSq = (robotsPositions[r, 0] - x_pos) ** 2 + (robotsPositions[r, 1] - y_pos) ** 2
                #distArray[r] = abs(math.sqrt(distanceSq))
                distArray[r] = abs(math.sqrt(distanceSq))
            minValue = np.min(distArray)
            minIndices = np.where(distArray == minValue)[0]
            for r in minIndices:
                locations_new[r].append([x_pos, y_pos])
                locationsIdx[r].append([i,j])
                robotDensity[r].append(densityArray[i,j])   



    Mass = np.zeros(robotsPositions.shape[0])
    C_x = np.zeros(robotsPositions.shape[0])
    C_y = np.zeros(robotsPositions.shape[0])
    locationalCost = 0
    
    for r in range(robotsPositions.shape[0]):
        Cx_r = 0
        Cy_r = 0
        Mass_r = 0
        locationInRobotRegion = np.array(locations_new[r])
        currentrobotLoc = robotsPositions[r]
        r_dens = robotDensity[r]  
        for pos in range(locationInRobotRegion.shape[0]):
            dens = r_dens[pos] #*resolution * resolution
            Mass_r += dens
            Cx_r += dens * locationInRobotRegion[pos, 0]
            Cy_r += dens * locationInRobotRegion[pos, 1]
            positionDiffSq = (locationInRobotRegion[pos, 0] - currentrobotLoc[0]) ** 2 + (locationInRobotRegion[pos, 1] - currentrobotLoc[1]) ** 2  
            # We are not integrating so this implementation includes resolution as well
            locationalCost += dens * (positionDiffSq)  #* resolution
        if(Mass_r!=0):
            Cx_r /= Mass_r
            Cy_r /= Mass_r
            C_x[r] = Cx_r
            C_y[r] = Cy_r
            Mass[r] = Mass_r
    '''        

            
    return C_x, C_y, locationalCost, Mass,hull_figHandles,text_handles,locationsIdx,locations_new
    
    
# Function to generate random means within the given domain
def generate_random_means(num_distributions, domain, rng):
    return rng.uniform(domain[0], domain[1], size=(num_distributions, 2))

# Function to generate random covariance matrix with the same variance
def generate_covariance_matrix(variance):
    return variance * np.identity(2)

# Function to calculate density function value for the entire function
def density_function(x, y, means, variances):
    num_distributions = len(means)
    result = 0
    for i in range(num_distributions):
        covariance_matrix = generate_covariance_matrix(variances[i])
        result += gaussian_density(x, y, means[i], covariance_matrix)
    return result

# Function to calculate density function value at a point (x, y) for a given Gaussian distribution
def gaussian_density(x, y, mean, covariance_matrix):
    rv = multivariate_normal(mean=mean, cov=covariance_matrix)
    return rv.pdf([x, y])

# plot mean and var using the given plot axes for pred_mean and pred_var
def plot_mean_and_var(X,Y,pred_mean,pred_var,pred_mean_fig=None,pred_var_fig=None):
    pred_mean_fig.clf()
    ax_2d_mean = pred_mean_fig.add_subplot()
    contour_plot_mean = ax_2d_mean.contourf(X, Y, pred_mean.reshape(X.shape[0],X.shape[1]), cmap='viridis')
    ax_2d_mean.set_title( 'predicted mean')
    ax_2d_mean.set_xlabel('X')
    ax_2d_mean.set_ylabel('Y')
    plt.colorbar(contour_plot_mean, ax= ax_2d_mean)

    pred_var_fig.clf()
    ax_2d_var = pred_var_fig.add_subplot()
    contour_plot_var = ax_2d_var.contourf(X, Y, pred_var.reshape(X.shape[0],X.shape[1]), cmap='Reds',vmin=0, vmax=1) #cmap='Reds'
    ax_2d_var.set_title( 'predicted std')
    ax_2d_var.set_xlabel('X')
    ax_2d_var.set_ylabel('Y')
    plt.colorbar(contour_plot_var, ax= ax_2d_var)
