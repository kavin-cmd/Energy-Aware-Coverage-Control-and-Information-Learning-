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
from GP import GP


##################################################################################
# Paper ref:
# [1] Maria Santos, Udari Madhushani, Alessia Benevento, and Naomi Ehrich Leonard. Multi-robot learning and coverage of unknown spatial fields. 
# In 2021 International Symposium on Multi-Robot and Multi-Agent Systems (MRS), pages 137–145. IEEE, 2021.
# Implemented by Aiman Munir - Ph.D. Candidate, UGA School of Computing
##################################################################################



def executeIPP_py(N=4, resolution=0.1, number_of_iterations=20, show_fig_flag=True, save_fig_flag=False):
    rng = np.random.default_rng(12345)
    distance_to_centroid_threshold= -0.1
    file_path = ""
    ROBOT_COLOR = {0: "red", 1: "green", 2: "blue", 3:"black",4:"grey",5:"orange",6:"cyan",7:"yellow",8:"magenta",9:"lime",10:"indigo"}
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1    
    # generate random initial values
    current_robotspositions =  rng.uniform(x_min, x_max, size=(2, N))
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

    # generate 9 Gauusian distribution for ground_truth
    # Using Z_phi here to represent ground_truth (phi(q))
    # Number of Gaussian distributions
    num_distributions = 1
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
    # for the 1st iteration setting the surrogate_mean
    surrogate_mean = copy.deepcopy(pred_mean)   
    beta_val = 0
    r_t = 0
    
    #########################################################  Main Code for Balancing Coverage and Informative Path Planning (IPP)
    goal_for_centroid = copy.deepcopy(current_robotspositions)
    sampling_goal = copy.deepcopy(current_robotspositions)
    for iteration in range(number_of_iterations):
        ## Get Robot Positions and Plotting
        positions_array[iteration,:,:] = current_robotspositions[:2,:]
        [main_axes.scatter(current_robotspositions[0,:], current_robotspositions[1,:], c="green", s=60, marker="x", linewidths=1) for i in range(N)]
        if iteration>0:
            [plot.remove() for plot in positions_plots]
        positions_plots = [main_axes.plot(positions_array[:iteration+1, 0, i], positions_array[:iteration+1, 1, i], color="green")[0] for i in range(N)]
    	# r_t for Eq 11 - control law from [1]
        if iteration > 0:
            r_t = 1/iteration
            discretization = ((X.shape[0])*(X.shape[1])) * math.pi* math.pi* iteration *iteration 
        else: 
            r_t = 1
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
        r_t = 0
        step_size = 1
        
        robotsPositions_diff = step_size  * (((1-r_t)*centroid)  - current_robotspositions[:2,:] + (r_t*sampling_goal) )
        
        
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
        
        #Set Velocities
        current_robotspositions[:2,:]  +=  robotsPositions_diff
        plt.pause(0.05)

    ## Saving All Plots
    if(show_fig_flag):
        # Fig: Area
        fig_area = plt.figure()
        ax_area = fig_area.add_subplot()
        ax_area.set_ylabel("Area")
        ax_area.set_xlabel("Iterations")
        area_plot = [ax_area.plot(iteration_array,areaArray[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_area.legend()

        # Fig: RMSE
        fig_rmse = plt.figure()
        ax_rmse = fig_rmse.add_subplot()
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_xlabel("Iterations")
        rmse_plot = ax_rmse.plot(iteration_array,rmse_array, color="black")
        
        # Fig: Variance
        fig_var = plt.figure()
        ax_rmse = fig_var.add_subplot()
        ax_rmse.set_ylabel("Variance")
        ax_rmse.set_xlabel("Iterations")
        rmse_plot = ax_rmse.plot(iteration_array,variance_array, color="black")

        # Fig: Cumulative Regret
        fig_regret = plt.figure()
        ax_regret = fig_regret.add_subplot()
        ax_regret.set_ylabel("Regret r(t)")
        ax_regret.set_xlabel("Iterations")
        regret_plot = ax_regret.plot(iteration_array,regret_array, color="black")

        # Fig: Beta Value of the Information Function (Surrogate Distribution)
        fig_beta_val = plt.figure()
        ax_beta_val = fig_beta_val.add_subplot()
        ax_beta_val.set_ylabel("Beta Value of GP-UCB")
        ax_beta_val.set_xlabel("Iterations")
        beta_val_plot = ax_beta_val.plot(iteration_array,beta_val_array)

        # Fig: Balancing Coefficient (gamma in Eq. 11 of [1] - VEC Approach)
        fig_rt_val = plt.figure()
        ax_rt_val = fig_rt_val.add_subplot()
        ax_rt_val.set_ylabel("Gamma Coefficient")
        ax_rt_val.set_xlabel("Iterations")
        rt_val_plot = ax_rt_val.plot(iteration_array,rt_array)
        
        # Fig: Centroid
        fig_dis_centroid = plt.figure()
        ax_dis_centroid = fig_dis_centroid.add_subplot()
        ax_dis_centroid.set_xlabel("Iterations")
        ax_dis_centroid.set_ylabel("Centroid Distance")
        centroid_plot = [ax_dis_centroid.plot(iteration_array,centroid_dist_array[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_dis_centroid.legend()
        
        # Fig: Cumulative Distance
        fig_cumulative_distance = plt.figure()
        ax_cumulative_dis = fig_cumulative_distance.add_subplot()
        ax_cumulative_dis.set_xlabel("Iterations")
        ax_cumulative_dis.set_ylabel("Cumulative Distance")
        cum_dis_plot = [ax_cumulative_dis.plot(iteration_array,cumulative_dist_array[:,i], color=ROBOT_COLOR[i], label=f'Robot {i+1}')[0] for i in range(N)]
        ax_cumulative_dis.legend()

        # Fig: Cumulative Distance
        fig_cost = plt.figure()
        ax_cost = fig_cost.add_subplot()
        ax_cost.set_xlabel("Iterations")
        ax_cost.set_ylabel("Locational Cost")
        cum_dis_plot = ax_cost.plot(iteration_array,locational_cost, color = "black",label="locationalCost")
        
        
        

        
        #saveFigs
        if save_fig_flag:
            fig_dis_centroid.savefig(file_path+"distance_to_centroid.png")
            fig_area.savefig(file_path+"area.png")
            fig_rmse.savefig(file_path+"rmse.png")
            fig_var.savefig(file_path+"variance.png")
            fig_cumulative_distance.savefig(file_path+"cumulative_distance.png")
            main_fig.savefig(file_path+"coverage.png")
            pred_mean_fig.savefig(file_path+"GPmean.png")
            pred_var_fig.savefig(file_path+"GPvar.png")
            fig_cost.savefig(file_path+"cost.png")
            fig_beta_val.savefig(file_path+"beta_val.png")
            fig_rt_val.savefig(file_path+"gamma_coeff.png")
            fig_regret.savefig(file_path+"regret.png")
            
        plt.show()
        plt.pause(20)
      
    return 


if __name__=="__main__":
    #Max 10 robots
    executeIPP_py(N=5, resolution=0.02,number_of_iterations=50, show_fig_flag=True,save_fig_flag=True)     
