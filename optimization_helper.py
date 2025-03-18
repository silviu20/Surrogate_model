import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
import datetime
from matplotlib.animation import FuncAnimation

def create_optimization_summary(optimization_file='optimization.csv'):
    """
    Create a summary of the optimization process with key statistics and save it to a file.
    """
    if not os.path.exists(optimization_file):
        print(f"Error: {optimization_file} not found!")
        return
    
    # Create summary directory if it doesn't exist
    summary_dir = Path('summary')
    summary_dir.mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(optimization_file)
    
    # Calculate key statistics
    stats = {
        'total_experiments': len(df),
        'best_yield': df['Yield'].max(),
        'best_yield_index': df['Yield'].idxmax(),
        'best_yield_params': df.loc[df['Yield'].idxmax()].to_dict(),
        'lowest_impurity': df['Impurity'].min(),
        'lowest_impurity_index': df['Impurity'].idxmin(),
        'lowest_impurity_params': df.loc[df['Impurity'].idxmin()].to_dict(),
        'best_impurity_ratio': df['ImpurityXRatio'].max(),
        'best_impurity_ratio_index': df['ImpurityXRatio'].idxmax(),
        'best_impurity_ratio_params': df.loc[df['ImpurityXRatio'].idxmax()].to_dict(),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Calculate a combined metric (this is just an example, adjust as needed)
    df['combined_score'] = df['Yield'] - df['Impurity'] + df['ImpurityXRatio']
    stats['best_combined_score'] = df['combined_score'].max()
    stats['best_combined_index'] = df['combined_score'].idxmax()
    stats['best_combined_params'] = df.loc[df['combined_score'].idxmax()].to_dict()
    
    # Save summary to file
    with open(summary_dir / 'optimization_summary.txt', 'w') as f:
        f.write(f"Optimization Summary (as of {stats['timestamp']})\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total experiments: {stats['total_experiments']}\n\n")
        
        f.write(f"Best Yield: {stats['best_yield']:.4f} (Experiment {stats['best_yield_index']+1})\n")
        f.write("Parameters:\n")
        for k, v in stats['best_yield_params'].items():
            if k not in ['Yield', 'Impurity', 'ImpurityXRatio', 'combined_score']:
                f.write(f"  {k}: {v:.4f}\n")
        f.write("\n")
        
        f.write(f"Lowest Impurity: {stats['lowest_impurity']:.4f} (Experiment {stats['lowest_impurity_index']+1})\n")
        f.write("Parameters:\n")
        for k, v in stats['lowest_impurity_params'].items():
            if k not in ['Yield', 'Impurity', 'ImpurityXRatio', 'combined_score']:
                f.write(f"  {k}: {v:.4f}\n")
        f.write("\n")
        
        f.write(f"Best Impurity Ratio: {stats['best_impurity_ratio']:.4f} (Experiment {stats['best_impurity_ratio_index']+1})\n")
        f.write("Parameters:\n")
        for k, v in stats['best_impurity_ratio_params'].items():
            if k not in ['Yield', 'Impurity', 'ImpurityXRatio', 'combined_score']:
                f.write(f"  {k}: {v:.4f}\n")
        f.write("\n")
        
        f.write(f"Best Combined Score: {stats['best_combined_score']:.4f} (Experiment {stats['best_combined_index']+1})\n")
        f.write("Parameters:\n")
        for k, v in stats['best_combined_params'].items():
            if k not in ['Yield', 'Impurity', 'ImpurityXRatio', 'combined_score']:
                f.write(f"  {k}: {v:.4f}\n")
    
    print(f"Summary saved to {summary_dir / 'optimization_summary.txt'}")
    return stats

def create_optimization_animations(optimization_file='optimization.csv'):
    """
    Create animations showing the progression of the optimization process
    """
    if not os.path.exists(optimization_file):
        print(f"Error: {optimization_file} not found!")
        return
    
    # Create animations directory if it doesn't exist
    anim_dir = Path('animations')
    anim_dir.mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(optimization_file)
    
    if len(df) < 3:  # Need at least 3 points for a meaningful animation
        print("Not enough data points for animation.")
        return
    
    # Create animation of convergence for each objective
    objectives = ['Yield', 'Impurity', 'ImpurityXRatio']
    objective_modes = ['max', 'min', 'max']  # Whether each objective should be maximized or minimized
    
    for obj, mode in zip(objectives, objective_modes):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def update(frame):
            ax.clear()
            
            data = df.iloc[:frame+1]
            
            # For plotting best-so-far
            if mode == 'max':
                best_so_far = data[obj].cummax()
            else:  # mode == 'min'
                best_so_far = data[obj].cummin()
            
            # Plot all points
            ax.plot(range(1, len(data)+1), data[obj], 'bo-', alpha=0.5, label='All Experiments')
            
            # Plot best-so-far
            ax.plot(range(1, len(data)+1), best_so_far, 'ro-', linewidth=2, label='Best So Far')
            
            ax.set_title(f'{obj} Convergence')
            ax.set_xlabel('Experiment Number')
            ax.set_ylabel(obj)
            ax.grid(True)
            ax.legend(loc='best')
            
            return ax,
        
        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(df), interval=500, blit=False)
        
        # Save the animation
        anim.save(anim_dir / f'{obj}_convergence.gif', writer='pillow', fps=2)
        plt.close(fig)
        
        print(f"Animation saved to {anim_dir / f'{obj}_convergence.gif'}")
    
    # Create 3D animation showing exploration of parameter space
    if len(df) >= 5:  # Need enough points for meaningful 3D visualization
        try:
            # Select the 3 most important parameters
            param_cols = [col for col in df.columns if col not in ['Yield', 'Impurity', 'ImpurityXRatio']]
            if len(param_cols) > 3:
                param_cols = param_cols[:3]  # Just use the first 3 for simplicity
            
            if len(param_cols) >= 3:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                def update(frame):
                    ax.clear()
                    
                    data = df.iloc[:frame+1]
                    
                    # Extract parameters for the 3D plot
                    x = data[param_cols[0]]
                    y = data[param_cols[1]]
                    z = data[param_cols[2]]
                    
                    # Color points based on yield
                    colors = data['Yield']
                    
                    # Plot points
                    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', 
                                        s=50, alpha=0.7, edgecolor='k')
                    
                    # Connect points in sequence
                    ax.plot(x, y, z, 'r-', alpha=0.3, linewidth=1)
                    
                    # Highlight the latest point
                    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], 
                               color='red', s=100, marker='*', label='Latest')
                    
                    # Set labels
                    ax.set_xlabel(param_cols[0])
                    ax.set_ylabel(param_cols[1])
                    ax.set_zlabel(param_cols[2])
                    ax.set_title(f'Parameter Space Exploration (Iter {frame+1})')
                    
                    # Add colorbar
                    if frame == 0:
                        fig.colorbar(scatter, ax=ax, label='Yield')
                    
                    return ax,
                
                # Create the animation
                anim = FuncAnimation(fig, update, frames=len(df), interval=500, blit=False)
                
                # Save the animation
                anim.save(anim_dir / 'parameter_space_exploration.gif', writer='pillow', fps=2)
                plt.close(fig)
                
                print(f"Animation saved to {anim_dir / 'parameter_space_exploration.gif'}")
        except Exception as e:
            print(f"Error creating 3D animation: {e}")

def archive_optimization_run(optimization_file='optimization.csv', archive_dir='optimization_archive'):
    """
    Archive the current optimization run with all associated data and visualizations
    """
    if not os.path.exists(optimization_file):
        print(f"Error: {optimization_file} not found!")
        return
    
    # Create archive directory if it doesn't exist
    archive_path = Path(archive_dir)
    archive_path.mkdir(exist_ok=True)
    
    # Create a timestamped subfolder for this archive
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_archive = archive_path / f"run_{timestamp}"
    run_archive.mkdir(exist_ok=True)
    
    # Copy optimization data
    shutil.copy2(optimization_file, run_archive / optimization_file)
    
    # Copy visualization files if they exist
    vis_dir = Path('visualization')
    if vis_dir.exists():
        vis_archive = run_archive / 'visualization'
        vis_archive.mkdir(exist_ok=True)
        for file in vis_dir.glob('*'):
            if file.is_file():
                shutil.copy2(file, vis_archive / file.name)
    
    # Copy animations if they exist
    anim_dir = Path('animations')
    if anim_dir.exists():
        anim_archive = run_archive / 'animations'
        anim_archive.mkdir(exist_ok=True)
        for file in anim_dir.glob('*'):
            if file.is_file():
                shutil.copy2(file, anim_archive / file.name)
    
    # Copy summary if it exists
    summary_dir = Path('summary')
    if summary_dir.exists():
        summary_archive = run_archive / 'summary'
        summary_archive.mkdir(exist_ok=True)
        for file in summary_dir.glob('*'):
            if file.is_file():
                shutil.copy2(file, summary_archive / file.name)
    
    # Create a README for this archive
    with open(run_archive / 'README.txt', 'w') as f:
        f.write(f"Optimization Run Archive\n")
        f.write(f"Created on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add some statistics about the run
        df = pd.read_csv(optimization_file)
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Best Yield: {df['Yield'].max():.4f}\n")
        f.write(f"Lowest Impurity: {df['Impurity'].min():.4f}\n")
        f.write(f"Best Impurity Ratio: {df['ImpurityXRatio'].max():.4f}\n")
    
    print(f"Optimization run archived to {run_archive}")
    return str(run_archive)

if __name__ == "__main__":
    # This code runs if the script is executed directly
    create_optimization_summary()
    create_optimization_animations()
    archive_optimization_run()