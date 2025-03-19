import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
import pandas as pd

def create_gp_surrogate_visualization(data, parameter_names, target_names, output_dir='visualization'):
    """
    Create visualizations of GP surrogate models for each parameter-target combination.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the optimization data
    parameter_names : list
        List of parameter column names to visualize
    target_names : list
        List of target column names to visualize
    output_dir : str
        Directory to save the plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of points to use for dense x-axis sampling
    n_points = 100
    
    # Create the surrogate model plots
    for target_name in target_names:
        for param_name in parameter_names:
            # Extract parameter and target values
            X = data[param_name].values.reshape(-1, 1)
            y = data[target_name].values
            
            # Standardize X and y for better GP performance
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            X_scaled = X_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
            
            # Create GP model with Matern kernel (better for noisy data)
            # We use a combination of a constant kernel and a Matern kernel
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
            
            # Fit the GP model
            gp.fit(X_scaled, y_scaled)
            
            # Create a dense grid of points for prediction
            x_min, x_max = X.min(), X.max()
            # Add some padding to the range
            x_pad = 0.05 * (x_max - x_min)
            x_dense = np.linspace(x_min - x_pad, x_max + x_pad, n_points).reshape(-1, 1)
            x_dense_scaled = X_scaler.transform(x_dense)
            
            # Get GP predictions and standard deviations
            y_pred_scaled, y_std_scaled = gp.predict(x_dense_scaled, return_std=True)
            
            # Transform back to original scale
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # For standard deviation, we need to scale it back properly
            # This is an approximation that works reasonably well
            y_std = y_std_scaled * y_scaler.scale_
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the GP mean prediction
            ax.plot(x_dense, y_pred, 'b-', label='GP estimate')
            
            # Plot the uncertainty bands (1 and 2 standard deviations)
            ax.fill_between(x_dense.ravel(), 
                           y_pred - 2*y_std, 
                           y_pred + 2*y_std, 
                           alpha=0.2, color='b')
            ax.fill_between(x_dense.ravel(), 
                           y_pred - y_std, 
                           y_pred + y_std, 
                           alpha=0.3, color='b', 
                           label='GP uncertainty')
            
            # Plot the actual data points with error bars
            # For error bars, we'll use the local standard deviation of nearby points
            window_size = (X.max() - X.min()) * 0.1  # 10% of the range
            error_bars = []
            for x_val in X:
                nearby_indices = np.where(np.abs(X - x_val) < window_size)[0]
                if len(nearby_indices) > 1:
                    local_std = np.std(y[nearby_indices])
                    error_bars.append(max(local_std, 0.05 * (y.max() - y.min())))
                else:
                    error_bars.append(0.05 * (y.max() - y.min()))
            
            ax.errorbar(X.ravel(), y, yerr=error_bars, fmt='ko', 
                      capsize=4, markersize=5, label='Data')
            
            # Find best point for this objective
            if target_name == "Impurity":  # For Impurity, lower is better
                best_idx = data[target_name].idxmin()
                best_val = data.loc[best_idx, param_name]
                ax.axvline(x=best_val, color='r', linestyle='--', label='Best value')
            else:  # For Yield and ImpurityXRatio, higher is better
                best_idx = data[target_name].idxmax()
                best_val = data.loc[best_idx, param_name]
                ax.axvline(x=best_val, color='r', linestyle='--', label='Best value')
            
            # Set labels and title
            ax.set_xlabel(param_name)
            ax.set_ylabel(target_name)
            ax.set_title(f'GP Surrogate Model: {target_name} vs {param_name}')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
            
            # Save the plot
            iteration = len(data)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/gp_surrogate_{target_name}_{param_name}_iter{iteration}.png')
            plt.savefig(f'{output_dir}/gp_surrogate_{target_name}_{param_name}_latest.png')
            plt.close(fig)
            
    print(f"GP surrogate model visualizations created in {output_dir}/")

def integrate_gp_surrogate_visualization(campaign_file='optimization.csv'):
    """
    Integrate GP surrogate visualization with existing optimization workflow.
    
    Parameters:
    -----------
    campaign_file : str
        Path to the optimization data file
    """
    # Load optimization data
    data = pd.read_csv(campaign_file)
    
    # Define parameter and target names
    parameter_names = ["T1Celsius", "t1min", "T2Celsius", "t2min", 
                     "EquivalentsReagent1", "EquivalentsBASE1"]
    target_names = ["Yield", "Impurity", "ImpurityXRatio"]
    
    # Create visualizations
    create_gp_surrogate_visualization(data, parameter_names, target_names)
    
    return "GP surrogate visualizations created successfully"

# If this script is run directly, create visualizations
if __name__ == "__main__":
    integrate_gp_surrogate_visualization()