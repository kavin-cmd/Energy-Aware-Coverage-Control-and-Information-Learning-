import rps.robotarium as Robotarium_
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from utilityFunctions_AOC_IPP import *
from GP import *
import matplotlib.patches as patches
from shapely.geometry import Point, LineString, Polygon


##################################################################################
# Paper ref:
# [1] Maria Santos, Udari Madhushani, Alessia Benevento, and Naomi Ehrich Leonard. Multi-robot learning and coverage of unknown spatial fields. 
# In 2021 International Symposium on Multi-Robot and Multi-Agent Systems (MRS), pages 137–145. IEEE, 2021.
# Implemented by Aiman Munir - Ph.D. Candidate, UGA School of Computing
##################################################################################

charger_colors = ['#00FF00', '#00B300', '#008000', '#00FF00', '#00FF80', '#008080']
battery_colors = ['#00FF00', '#00B300', '#008000', '#00FF00', '#00FF80', '#008080', '#00FF00', '#808000', '#008000', '#FF00FF']

recharger1_point = patches.Circle((-1, -1), radius=0.03, color=charger_colors[4])  # Blue color for charging points
recharger2_point = patches.Circle((1, -1), radius=0.03, color=charger_colors[4])
recharger3_point = patches.Circle((-1, 1), radius=0.03, color=charger_colors[4])
recharger4_point = patches.Circle((1, 1), radius=0.03, color=charger_colors[4])
recharger5_point = patches.Circle((0, 0.5), radius=0.03, color=charger_colors[4])
recharger6_point = patches.Circle((0.5, 0), radius=0.03, color=charger_colors[4])

recharger_points = [recharger1_point, recharger2_point, recharger3_point, recharger4_point, recharger5_point]



# Battery threshold
battery_threshold = 4.0  # Below 40%

idle_consumption_rate = 0.01
moving_consumption_rate = 0.15

# Assign charging points to robots
robot_charging_points = [recharger1_point, recharger2_point, recharger3_point, recharger4_point, recharger5_point, recharger6_point]

max_detection_range = 0.05

def find_nearest_charging_station(robot_position, charging_points):
    min_distance = float('inf')
    nearest_station = None
    for station in charging_points:
        station_center = np.array(station.center)  # Extract center coordinates
        distance = np.linalg.norm(robot_position - station_center)  # Calculate distance
        if distance < min_distance:
            min_distance = distance
            nearest_station = station_center
    return nearest_station

# Define obstacle points similar to recharging stations
obstacle1_point = patches.Rectangle((-0.8, -0.2), 0.3, 0.3, color='gray', alpha=0.5)
obstacle2_point = patches.Rectangle((0.2, 0.2), 0.2, 0.2, color='gray', alpha=0.5)
obstacle3_point = patches.Circle((0, -0.8), radius=0.03, color='gray', alpha=0.5)
obstacle4_point = patches.Circle((0.8, 0), radius=0.05, color='gray', alpha=0.5)
obstacle5_point = patches.Rectangle((-0.4, 0.6), 0.8, 0.2, color='gray', alpha=0.5)

# Add obstacle points to the list of obstacles
obstacle_points = [obstacle1_point, obstacle2_point, obstacle3_point, obstacle4_point, obstacle5_point]

def check_surroundings(current_position, max_detection_range, obstacles):
    detected_obstacles = []
    current_position_point = Point(current_position)
    
    for obstacle in obstacles:
        # Convert obstacle to Shapely geometry
        if isinstance(obstacle, patches.Rectangle):
            obstacle_geometry = LineString([(obstacle.get_x(), obstacle.get_y()), 
                                            (obstacle.get_x() + obstacle.get_width(), obstacle.get_y()), 
                                            (obstacle.get_x() + obstacle.get_width(), obstacle.get_y() + obstacle.get_height()), 
                                            (obstacle.get_x(), obstacle.get_y() + obstacle.get_height()), 
                                            (obstacle.get_x(), obstacle.get_y())])
        elif isinstance(obstacle, patches.Circle):
            center = obstacle.center
            radius = obstacle.get_radius()
            obstacle_geometry = Point(center).buffer(radius)
        else:
            raise ValueError("Unsupported obstacle type")
        
        # Check if the obstacle is within detection range
        if current_position_point.distance(obstacle_geometry) <= max_detection_range:
            detected_obstacles.append(obstacle)
    
    return detected_obstacles


def executeIPP_py(N=4, resolution=0.1, number_of_iterations=20, show_fig_flag=True, save_fig_flag=False):
    battery_levels = np.full(N,10.0)
    # battery_levels[2:4] = 5.0
    alpha = 0.1  # Energy decay factor
    beta = 0.5  # Distance decay factor

    rng = np.random.default_rng(12345)
    distance_to_centroid_threshold= -0.1
    file_path = ""
    ROBOT_COLOR = {0: "red", 1: "green", 2: "blue", 3:"black",4:"grey",5:"orange",6:"cyan",7:"yellow",8:"magenta",9:"lime",10:"indigo"}
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1    
    # generate random initial values
    # current_robotspositions =  rng.uniform(x_min, x_max, size=(2, N))
    current_robotspositions = np.array([
    [recharger1_point.center[0], recharger2_point.center[0], recharger3_point.center[0], recharger4_point.center[0], recharger5_point.center[0], recharger6_point.center[0]],
    [recharger1_point.center[1], recharger2_point.center[1], recharger3_point.center[1], recharger4_point.center[1], recharger5_point.center[1], recharger6_point.center[1]]
])
    # to show interactive plotting
    plt.ion()
    main_fig = plt.figure()
    main_axes = main_fig.add_subplot()
    pred_mean_fig = plt.figure()
    pred_var_fig = plt.figure()
    boundary_points = [[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]] 
    bound_x, bound_y = zip(*boundary_points) 
    cumulative_dist_array = np.zeros((number_of_iterations,N))
    centroid_dist_array = np.zeros((number_of_iterations,N))
    areaArray = np.zeros((number_of_iterations,N))
    cumulative_distance = np.zeros(N)
    rmse_array = np.zeros(number_of_iterations)
    variance_array = np.zeros(number_of_iterations)
    positions_array = np.zeros((number_of_iterations,2,N))
    positions_last_timeStep = np.zeros((2,N)) 
    regret_array = np.ones(number_of_iterations)
    iteration_array = np.zeros(number_of_iterations)
    beta_val_array = np.ones(number_of_iterations)
    rt_array = np.ones(number_of_iterations)
    locational_cost = np.ones(number_of_iterations)
    current_position_marker_handle = []
    charging_station = [recharger1_point.center[0], recharger1_point.center[1]] 
    charging_threshold = 0.05
    charging_rate = 0.1
    transition_iterations = 10
    battery_levels_array = np.zeros((number_of_iterations, N))

    # the obstacles are within the visible range of the axes
    main_axes.set_xlim([-1.0, 1.0])
    main_axes.set_ylim([-1.0, 1.0])

    # Create a plot for battery levels
    plt.figure()
    plt.xlabel('Robot')
    plt.ylabel('Battery Level')
    plt.title('Battery Levels')

    # Add a horizontal line for the battery threshold
    plt.axhline(y=battery_threshold, color='r', linestyle='--', label='Battery Threshold')

    # Assign charging points to robots
    robot_charging_points = [recharger1_point, recharger2_point, recharger3_point, recharger4_point, recharger5_point, recharger6_point]

    # generate 9 Gauusian distribution for ground_truth
    # Using Z_phi here to represent ground_truth (phi(q))
    # Number of Gaussian distributions
    num_distributions = 1 # number of gaussian distributions in each robot's position 3 for trimodel
    # Variances for all distributions
    variances = np.ones(num_distributions) * 0.05  # Adjusted variance for visibility
    # Generate random means for both density functions
    means_phi = generate_random_means(num_distributions, (x_min,x_max), rng)
    # Create a grid of points for plotting
    x_vals = np.arange(x_min, x_max + resolution, resolution) 
    y_vals = np.arange(y_min, y_max + resolution, resolution)
    X = np.zeros((len(x_vals), len(y_vals)))
    Y = np.zeros((len(x_vals), len(y_vals)))
    # Fill X and Y arrays using for loops
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            X[i, j] = x
            Y[i, j] = y
    Z_phi = np.vectorize(lambda x, y: density_function(x, y, means_phi, variances))(X, Y)

    test_X = np.column_stack((X.flatten(),Y.flatten()))
    # plot the density to main fig to get an idea of where robots are going
    # can comment this line to turn off
    main_axes.pcolor(X, Y, Z_phi) #,shading="auto")
    train_X = np.transpose(current_robotspositions)
    Z_phi_current_pos = np.vectorize(lambda x, y: density_function(x, y, means_phi, variances))(current_robotspositions[0,:], current_robotspositions[1,:])
    train_Y = Z_phi_current_pos.reshape(train_X.shape[0],1)
    model = GP(sigma_f=np.var(Z_phi))
    model.train(train_X,train_Y)
    pred_mean, pred_var = model.predict(test_X)
    bar_plot = plt.bar(range(N), battery_levels, color=battery_colors)
    # for the 1st iteration setting the surrogate_mean
    surrogate_mean = copy.deepcopy(pred_mean)   
    beta_val = 0
    r_t = 0.0
    
    #########################################################  Main Code for Balancing Coverage and Informative Path Planning (IPP)
    goal_for_centroid = copy.deepcopy(current_robotspositions)
    sampling_goal = copy.deepcopy(current_robotspositions)
    min_energy_array = np.zeros(number_of_iterations)
    max_energy_array = np.zeros(number_of_iterations)
    mean_energy_array = np.zeros(number_of_iterations)


    for iteration in range(number_of_iterations):
        # Calculate minimum, maximum, and mean energy across robots
        min_energy = np.min(battery_levels)
        max_energy = np.max(battery_levels)
        mean_energy = np.mean(battery_levels)
        
        # Store the values in arrays
        min_energy_array[iteration] = min_energy
        max_energy_array[iteration] = max_energy
        mean_energy_array[iteration] = mean_energy
        ## Get Robot Positions and Plotting
        positions_array[iteration,:,:] = current_robotspositions[:2,:]
        for recharger_point in recharger_points:
            main_axes.add_patch(recharger_point)

        # Add obstacle points to the main figure
        for obstacle_point in obstacle_points:
            main_axes.add_patch(obstacle_point)
        [main_axes.scatter(current_robotspositions[0,:], current_robotspositions[1,:], c="green", s=60, marker="x", linewidths=1) for i in range(N)]
        if iteration>0:
            [plot.remove() for plot in positions_plots]
        positions_plots = [main_axes.plot(positions_array[:iteration+1, 0, i], positions_array[:iteration+1, 1, i], color="green")[0] for i in range(N)]
    	# r_t for Eq 11 - control law from [1]
        if iteration > 0:
            r_t = 1/iteration
            discretization = ((X.shape[0])*(X.shape[1])) * math.pi* math.pi* iteration *iteration 
        else: 
            r_t = 1.0
            discretization = ((X.shape[0])*(X.shape[1])) * math.pi* math.pi
        print(f"iteration:{iteration}")
        iteration_array[iteration] = iteration    

        # remove boundaries for previous voronoi partition
        # before plotting on top of
        if(iteration>0):
            for robot_r in range(len(global_hull_figHandles)):
                current_position_marker_handle[robot_r].remove()
                hullObject = global_hull_figHandles[robot_r]
                hullObject.remove()

        battery_levels_array[iteration, :] = battery_levels
        
        ## Coverage Part of the Algorithm
        # Perform Voronoi Partitioning        
        current_position_marker_handle = [main_axes.scatter(current_robotspositions[0,:], current_robotspositions[1,:], edgecolors="red", facecolors='none', s=100, marker="o", linewidths=3) for i in range(N)]
        main_axes.set_xlim([-1.2,1.2])
        main_axes.set_ylim([-1.2,1.2])
        
        # Get the centroid of the Voronoi regions weighted by the surrogate function (pred_mean)
        # Using the Partition Finder utility
        C_x, C_y , cost, area, global_hull_figHandles, global_hull_textHandles,locationIdx,locationx = partitionFinder(main_axes,np.transpose(current_robotspositions[:2,:N]), [x_min,x_max], [y_min,y_max], resolution, surrogate_mean.reshape(Z_phi.shape), Z_phi) 
        centroid = (np.array([C_x,C_y]))
        
        ## Learning/IPP Part of the Algorithm
        # Start GPR process for learning the surrogate model of the underlying (unknown) spatial distribution phi(x)
        train_X = np.transpose(current_robotspositions)
        Z_phi_current_pos = np.vectorize(lambda x, y: density_function(x, y, means_phi, variances))(current_robotspositions[0,:], current_robotspositions[1,:])
        # Beta for Eq 9 - surrogate_mean from [1] for GP-UCB
        delta_UCB = 0.1
        beta_val = 2*math.log( discretization/(6*delta_UCB))
        beta_val_array[iteration] = beta_val
        rt_array[iteration] = r_t
        train_Y = Z_phi_current_pos.reshape(train_X.shape[0],1)
        #if iteration>0:
        model.train(train_X,train_Y)
        pred_mean, pred_var = model.predict(test_X)
        pred_var = pred_var.reshape(X.shape[0],X.shape[1])
        pred_std = np.sqrt(pred_var)
        plot_mean_and_var(X,Y,pred_mean,pred_std.reshape(pred_mean.shape),pred_mean_fig=pred_mean_fig,pred_var_fig=pred_var_fig)
        
        # Eq 9 from paper [1]
        # phi^(t)(q) = mu^(t-1)(q) - sqrt(beta^(t)) * sigma^(t-1)(q), for all q in D
        #surrogate_mean = copy.deepcopy(pred_mean) - (math.sqrt(beta_val)*pred_std.reshape(pred_mean.shape))
        surrogate_mean = (pred_mean) + (math.sqrt(beta_val)*pred_std.reshape(pred_mean.shape)) 
        
        # Get the locations for each robot within their Voronoi regions, where the variance is maximum (VEC approach)
        for robot in range(N):
            location_ids = np.array(locationIdx[robot])
            locations = np.array(locationx[robot])
            if len(location_ids) != 0:
                std_in_voronoi_region = (pred_std[location_ids[:, 0], location_ids[:, 1]])
                idx_with_max_std = np.argmax(std_in_voronoi_region)        
                if robot==0:
                    sampling_goal[:,robot] = locations[idx_with_max_std] 
                else:   
                    # a quick fix for robots getting same position for sampling
                    # it can happen because they share same voronoi boundary
                    # check with other robots if they have similar positions
                    # In case of similar positions, assign a larger negative value to that standard deviation and then recalculate the position
                    similar_goal = True
                    while similar_goal:
                        sampling_goal[:,robot] = locations[idx_with_max_std] 
                        similar_goal = np.any(np.all(sampling_goal[:,:robot] == (sampling_goal[:,robot]).reshape(2,1), axis=0))
                        if similar_goal==False:
                            break           
                        std_in_voronoi_region[idx_with_max_std] = -10000
                        idx_with_max_std = np.argmax(std_in_voronoi_region)
        #centroid = (np.array([C_x,C_y]))
        
      	## Coverage Controller with Learning Objective (Calculating Velcotiy u_i(t) for each robot)
	    # Eq. 11 from [1]
        # Control law to find x_i dot : \dot{x}_i^(t) = (1 - gamma) * c_i^(t) + gamma * e_i^(t), for i in {1, ..., N}. Here r_t variable is the gamma(t)
        # r_t = 0
        # Calculating robots' positions difference
        step_size = 1.0
        robotsPositions_diff = np.zeros((2, N))
        for robot in range(N):
            # Find nearest charging station for the current robot
            nearest_charging_station = find_nearest_charging_station(current_robotspositions[:2, robot], robot_charging_points)

            # Calculate distance to charging station
            dist_to_cs = np.linalg.norm(current_robotspositions[:2, robot] - nearest_charging_station)

            # Calculate energy consumption for movement
            distance_traveled = np.linalg.norm(robotsPositions_diff[:, robot])
            energy_consumption = moving_consumption_rate * distance_traveled

            # Apply energy decay for the robot's battery level
            battery_levels[robot] -= energy_consumption + alpha * 1 + beta * distance_traveled
            # Check if the robot is close enough to the charging station to recharge
            if np.linalg.norm(current_robotspositions[:2, robot] - nearest_charging_station) < charging_threshold:
                # Recharge the battery to 100%
                battery_levels[robot] = 10.0
            # Check if the battery level is below the threshold
            if battery_levels[robot] < battery_threshold:
                # Move towards charging station gradually
                charging_transition = current_robotspositions[:2, robot] + (nearest_charging_station - current_robotspositions[:2, robot]) * (1 / transition_iterations)
                robotsPositions_diff[:, robot] = step_size * (charging_transition - current_robotspositions[:2, robot])
            else:
                # Move towards sampling goal gradually
                sampling_transition = current_robotspositions[:2, robot] + (sampling_goal[:, robot] - current_robotspositions[:2, robot]) * (1 / transition_iterations)
                robotsPositions_diff[:, robot] = step_size * (sampling_transition - current_robotspositions[:2, robot])

            bar_plot[robot].set_height(battery_levels[robot])
            bar_plot[robot].set_color(battery_colors[robot] if battery_levels[robot] >= battery_threshold else 'red')
            plt.draw()
            plt.pause(0.05)

            detected_obstacles = check_surroundings(current_robotspositions[:, robot], max_detection_range, obstacle_points)

            if detected_obstacles:
                print(f"Robot {robot+1} detected obstacles:", detected_obstacles)

        ## Calculating All Performance Metrics
        locational_cost[iteration] = cost       
        #Calculate cumulative regret
        if iteration>0:
            regret_array[iteration] = cost - np.min(locational_cost[:iteration])
            
        dist_to_centroid = np.ones((N))
        for robot in range(N):
            areaArray[iteration,robot] = area[robot]
            dist_to_centroid[robot] = (math.sqrt((current_robotspositions[ 0,robot] - C_x[robot]) ** 2 + (current_robotspositions[1,robot] - C_y[robot]) ** 2))
            dist_to_centroid[robot] =  round(dist_to_centroid[robot], 2)
            centroid_dist_array[iteration,robot] = dist_to_centroid[robot]
            # find goal for balancing both centroid and sampling 
            # using  X(t+1) = x(t) + \dot{x}_i
            goal_for_centroid[:,robot] = (current_robotspositions[:2,robot] + np.round(robotsPositions_diff[:2,robot],decimals=2))
            if iteration>0:
                d = dist(positions_last_timeStep[0,robot], positions_last_timeStep[1,robot], (current_robotspositions[0,robot],current_robotspositions[1,robot]))
                cumulative_distance[robot] = cumulative_distance[robot] + d
                cumulative_dist_array[iteration,robot]=cumulative_distance[robot]  
        # Equation: RMSE = sqrt(mean((y_true - y_pred)**2)) : Equivalent to the Density error
        rmse = np.sqrt(np.mean(np.square(Z_phi.flatten() - pred_mean.flatten()))) # getting the rmse of the error across the entire map
        rmse_array[iteration] = rmse
        
        variance_metric = np.max(pred_var.flatten()) # Getting the max variance across the entire map
        variance_array[iteration] = variance_metric
        positions_last_timeStep = copy.deepcopy(current_robotspositions)
        
        # Currently the next positions of robots are calculated using sampling goal only - 
        # based on max std in robot's current partition
        #current_robotspositions = copy.deepcopy(sampling_goal) 
        
        # Update robot positions using calculated velocity
        current_robotspositions[:2, :] += robotsPositions_diff

        plt.pause(0.05)

    ## Saving All Plots
    if(show_fig_flag):
        # # Fig: Area
        # fig_area = plt.figure()
        # ax_area = fig_area.add_subplot()
        # ax_area.set_ylabel("Area")
        # ax_area.set_xlabel("Iterations")
        # area_plot = [ax_area.plot(iteration_array,areaArray[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        # ax_area.legend()

        # # Fig: RMSE
        # fig_rmse = plt.figure()
        # ax_rmse = fig_rmse.add_subplot()
        # ax_rmse.set_ylabel("RMSE")
        # ax_rmse.set_xlabel("Iterations")
        # rmse_plot = ax_rmse.plot(iteration_array,rmse_array, color="black")
        
        # # Fig: Variance
        # fig_var = plt.figure()
        # ax_rmse = fig_var.add_subplot()
        # ax_rmse.set_ylabel("Variance")
        # ax_rmse.set_xlabel("Iterations")
        # rmse_plot = ax_rmse.plot(iteration_array,variance_array, color="black")

        # # Fig: Cumulative Regret
        # fig_regret = plt.figure()
        # ax_regret = fig_regret.add_subplot()
        # ax_regret.set_ylabel("Regret r(t)")
        # ax_regret.set_xlabel("Iterations")
        # regret_plot = ax_regret.plot(iteration_array,regret_array, color="black")

        # # Fig: Beta Value of the Information Function (Surrogate Distribution)
        # fig_beta_val = plt.figure()
        # ax_beta_val = fig_beta_val.add_subplot()
        # ax_beta_val.set_ylabel("Beta Value of GP-UCB")
        # ax_beta_val.set_xlabel("Iterations")
        # beta_val_plot = ax_beta_val.plot(iteration_array,beta_val_array)

        # # Fig: Balancing Coefficient (gamma in Eq. 11 of [1] - VEC Approach)
        # fig_rt_val = plt.figure()
        # ax_rt_val = fig_rt_val.add_subplot()
        # ax_rt_val.set_ylabel("Gamma Coefficient")
        # ax_rt_val.set_xlabel("Iterations")
        # rt_val_plot = ax_rt_val.plot(iteration_array,rt_array)
        
        # # Fig: Centroid
        # fig_dis_centroid = plt.figure()
        # ax_dis_centroid = fig_dis_centroid.add_subplot()
        # ax_dis_centroid.set_xlabel("Iterations")
        # ax_dis_centroid.set_ylabel("Centroid Distance")
        # centroid_plot = [ax_dis_centroid.plot(iteration_array,centroid_dist_array[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        # ax_dis_centroid.legend()
        
        # # Fig: Cumulative Distance
        # fig_cumulative_distance = plt.figure()
        # ax_cumulative_dis = fig_cumulative_distance.add_subplot()
        # ax_cumulative_dis.set_xlabel("Iterations")
        # ax_cumulative_dis.set_ylabel("Cumulative Distance")
        # cum_dis_plot = [ax_cumulative_dis.plot(iteration_array,cumulative_dist_array[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        # ax_cumulative_dis.legend()

        # # Fig: Cumulative Distance
        # fig_cost = plt.figure()
        # ax_cost = fig_cost.add_subplot()
        # ax_cost.set_xlabel("Iterations")
        # ax_cost.set_ylabel("Locational Cost")
        # cum_dis_plot = ax_cost.plot(iteration_array,locational_cost, color = "black",label="locationalCost")

        # Fig: Battery Levels
        fig_battery = plt.figure()
        ax_battery = fig_battery.add_subplot()
        ax_battery.set_xlabel("Iterations")
        ax_battery.set_ylabel("Battery Level")
        for i in range(N):
            ax_battery.plot(iteration_array, battery_levels_array[:, i], label=f'Robot {i+1}', color=ROBOT_COLOR[i])
        ax_battery.legend()

        fig_battery_mmm = plt.figure()
        plt.plot(iteration_array, min_energy_array, label='Min Energy', linestyle='-', color='blue')
        plt.plot(iteration_array, max_energy_array, label='Max Energy', linestyle='--', color='green')
        plt.plot(iteration_array, mean_energy_array, label='Mean Energy', linestyle=':', color='red')
        plt.xlabel('Iterations')
        plt.ylabel('Energy Level')
        plt.title('Energy Levels Across Iterations')
        plt.legend()
        plt.grid(True)

        #saveFigs
        if save_fig_flag:
            # fig_dis_centroid.savefig(file_path+"distance_to_centroid.png")
            # fig_area.savefig(file_path+"area.png")
            # fig_rmse.savefig(file_path+"rmse.png")
            # fig_var.savefig(file_path+"variance.png")
            # fig_cumulative_distance.savefig(file_path+"cumulative_distance.png")
            main_fig.savefig(file_path+"coverage.png")
            pred_mean_fig.savefig(file_path+"GPmean.png")
            pred_var_fig.savefig(file_path+"GPvar.png")
            # fig_cost.savefig(file_path+"cost.png")
            # fig_beta_val.savefig(file_path+"beta_val.png")
            # fig_rt_val.savefig(file_path+"gamma_coeff.png")
            # fig_regret.savefig(file_path+"regret.png")
            fig_battery.savefig('battery_levels_plot.png')
            fig_battery_mmm.savefig('battery_energy_plots.png')
            
        plt.show()
        plt.pause(20)
      
    return 


if __name__=="__main__":
    #Max 10 robots
    executeIPP_py(N=6, resolution=0.02,number_of_iterations=100, show_fig_flag=True,save_fig_flag=True)     
