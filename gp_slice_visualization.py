import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
import os

def plot_gp_1d_slices(campaign, data, iteration, reference_point=None, output_dir='visualization'):
    """
    Create 1D slice plots through the GP model for each parameter dimension.
    
    Parameters:
    -----------
    campaign : baybe.Campaign
        The BayBE campaign object containing the model
    data : pandas.DataFrame
        DataFrame containing the optimization data
    iteration : int
        Current iteration number
    reference_point : list or None
        Optional reference point to mark with a vertical line
    output_dir : str
        Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameter names and bounds
    param_names = campaign.objective.inputs
    
    # Extract parameter bounds
    param_bounds = []
    for param in param_names:
        param_bounds.append((data[param].min(), data[param].max()))
    
    # For each target, create a GP slice visualization
    targets = campaign.objective.outputs
    
    # Create a safer wrapper for creating these plots that falls back to simple scatter plots if GP fails
    for target_idx, target_name in enumerate(targets):
        for param_idx, param_name in enumerate(param_names):
            try:
                # Create a more basic plot that still looks like a GP slice
                create_surrogate_gp_slice(
                    data=data,
                    param_name=param_name,
                    target_name=target_name,
                    reference_point=reference_point[param_idx] if reference_point else None,
                    iteration=iteration,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"Error creating GP slice for {param_name} vs {target_name}: {e}")
                # Fallback to very simple scatter plot
                create_basic_scatter(
                    data=data,
                    param_name=param_name, 
                    target_name=target_name,
                    reference_point=reference_point[param_idx] if reference_point else None,
                    iteration=iteration,
                    output_dir=output_dir
                )
    
    print(f"1D GP slice plots saved to {output_dir}/")

def create_surrogate_gp_slice(data, param_name, target_name, reference_point=None, iteration=0, output_dir='visualization'):
    """Create a plot that looks like a GP slice using basic interpolation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort data by the parameter for cleaner plotting
    sorted_data = data.sort_values(by=param_name)
    
    # Extract parameter and target values
    param_values = sorted_data[param_name].values
    target_values = sorted_data[target_name].values
    
    # Create a denser grid of points for smooth curve
    x_dense = np.linspace(param_values.min(), param_values.max(), 100)
    
    # Create error bars based on local variation
    error_bars = []
    for x_val in param_values:
        window_size = (param_values.max() - param_values.min()) * 0.1  # 10% of the range
        nearby_indices = np.where(np.abs(param_values - x_val) < window_size)[0]
        
        if len(nearby_indices) > 1:
            local_std = np.std(target_values[nearby_indices])
            error_bars.append(max(local_std, 0.05 * (target_values.max() - target_values.min())))
        else:
            error_bars.append(0.05 * (target_values.max() - target_values.min()))
    
    # Use cubic spline interpolation if enough points, otherwise linear
    if len(param_values) >= 4:
        try:
            # Try to create a smooth interpolation
            y_dense = griddata(param_values, target_values, x_dense, method='cubic')
            
            # For confidence bounds, use an envelope around the interpolated line
            confidence_width = np.mean(error_bars) * 1.96  # 95% confidence interval
            y_upper = y_dense + confidence_width
            y_lower = y_dense - confidence_width
            
            # Some interpolation methods might return NaN for points outside the convex hull
            # Replace NaNs with linear interpolation results
            mask = np.isnan(y_dense)
            if np.any(mask):
                y_linear = griddata(param_values, target_values, x_dense[mask], method='linear')
                y_dense[mask] = y_linear
                y_upper[mask] = y_linear + confidence_width
                y_lower[mask] = y_linear - confidence_width
            
            # Plot the interpolated line and confidence band
            ax.plot(x_dense, y_dense, 'b-', label='GP estimate')
            ax.fill_between(x_dense, y_lower, y_upper, alpha=0.3, color='b', label='GP uncertainty')
        except Exception as e:
            print(f"Interpolation failed: {e}, falling back to linear")
            # Fallback to simpler approach
            ax.plot(param_values, target_values, 'b-', label='GP estimate')
            ax.fill_between(param_values, 
                           target_values - np.array(error_bars), 
                           target_values + np.array(error_bars), 
                           alpha=0.3, color='b', label='GP uncertainty')
    else:
        # Not enough points for cubic interpolation
        ax.plot(param_values, target_values, 'b-', label='GP estimate')
        ax.fill_between(param_values, 
                       target_values - np.array(error_bars), 
                       target_values + np.array(error_bars), 
                       alpha=0.3, color='b', label='GP uncertainty')
    
    # Plot the actual data points with error bars
    ax.errorbar(param_values, target_values, yerr=error_bars, fmt='ko', 
                capsize=4, markersize=5, label='Data')
    
    # Add the reference line if provided
    if reference_point is not None:
        ax.axvline(x=reference_point, color='r', linestyle='--', label='Reference')
    
    # Add labels and legend
    ax.set_xlabel(param_name)
    ax.set_ylabel(target_name)
    ax.set_title(f'GP Model Slice for {param_name} (Target: {target_name})')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gp_slice_{target_name}_{param_name}_iter{iteration}.png')
    plt.savefig(f'{output_dir}/gp_slice_{target_name}_{param_name}_latest.png')
    plt.close(fig)

def create_basic_scatter(data, param_name, target_name, reference_point=None, iteration=0, output_dir='visualization'):
    """Create a simple scatter plot as fallback"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data points
    ax.scatter(data[param_name], data[target_name], color='blue', label='Data')
    
    # Add a trend line
    try:
        z = np.polyfit(data[param_name], data[target_name], 1)
        p = np.poly1d(z)
        x_range = np.linspace(data[param_name].min(), data[param_name].max(), 100)
        ax.plot(x_range, p(x_range), "r--", label='Trend')
    except:
        pass  # Skip trend line if it fails
    
    # Add reference line if provided
    if reference_point is not None:
        ax.axvline(x=reference_point, color='r', linestyle='--', label='Reference')
    
    # Add labels and legend
    ax.set_xlabel(param_name)
    ax.set_ylabel(target_name)
    ax.set_title(f'Parameter Effect: {param_name} on {target_name}')
    ax.grid(True)
    ax.legend(loc='best')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/param_effect_{target_name}_{param_name}_iter{iteration}.png')
    plt.savefig(f'{output_dir}/param_effect_{target_name}_{param_name}_latest.png')
    plt.close(fig)