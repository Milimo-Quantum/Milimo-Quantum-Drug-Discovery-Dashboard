# visualization.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import io
import base64
import logging
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import MDAnalysis as mda

logger = logging.getLogger(__name__)

def generate_molecular_docking_visualization(vqe_result, expectation_values, optimized_params, file_path):
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Use MDAnalysis to handle both PDB and MOL2 files
        universe = mda.Universe(file_path)
        atom_coords = universe.atoms.positions

        # Plot protein/molecule structure
        ax.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2], c='blue', s=10, alpha=0.5)

        # Generate ligand positions based on optimized parameters
        num_params = len(optimized_params)
        num_coords = num_params // 3  # Ensure we have complete x, y, z coordinates
        
        ligand_x = optimized_params[:num_coords]
        ligand_y = optimized_params[num_coords:2*num_coords]
        ligand_z = optimized_params[2*num_coords:3*num_coords]

        # Ensure all arrays have the same length
        min_length = min(len(ligand_x), len(ligand_y), len(ligand_z))
        ligand_x = ligand_x[:min_length]
        ligand_y = ligand_y[:min_length]
        ligand_z = ligand_z[:min_length]

        # Plot ligand
        ax.scatter(ligand_x, ligand_y, ligand_z, c='red', s=100, label='Ligand')

        # Calculate and plot the binding site (centroid of the ligand)
        binding_site = (np.mean(ligand_x), np.mean(ligand_y), np.mean(ligand_z))
        ax.scatter(*binding_site, c='yellow', s=200, label='Binding Site')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Molecular Docking Visualization\nVQE Energy: {vqe_result.optimal_value:.4f}')

        ax.legend()

        # Save the plot to a buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)

        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error in generate_molecular_docking_visualization: {str(e)}")
        raise

def create_quantum_state_distribution_chart(counts):
    try:
        states = list(counts.keys())
        probabilities = list(counts.values())
        total_probability = sum(probabilities)
        normalized_probabilities = [p / total_probability * 100 for p in probabilities]

        # Convert states to binary and pad to ensure equal length
        max_length = max(len(s) for s in states)
        binary_states = [s.zfill(max_length) for s in states]

        # Create a 2D grid of probabilities
        grid_size = int(np.ceil(np.sqrt(len(binary_states))))
        grid = np.zeros((grid_size, grid_size))
        for i, prob in enumerate(normalized_probabilities):
            row = i // grid_size
            col = i % grid_size
            grid[row, col] = prob

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid, cmap='viridis')
        
        ax.set_title('Quantum State Distribution Heatmap')
        ax.set_xlabel('Quantum State (X)')
        ax.set_ylabel('Quantum State (Y)')
        
        # Remove tick labels
        ax.set_xticks([])
        ax.set_yticks([])

        plt.colorbar(im, ax=ax, label='Probability (%)')

        # Save the plot to a buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)

        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error in create_quantum_state_distribution_chart: {str(e)}")
        raise

def create_drug_discovery_pipeline_visualization():
    try:
        steps = ['PDB Input', 'Quantum Sim', 'AI Analysis', 'ADME Prediction', 'Lead Optimization', 'Preclinical', 'Clinical Trials']
        num_steps = len(steps)
        
        fig, ax = plt.subplots(figsize=(14, 2))
        ax.set_xlim(0, num_steps)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Positioning parameters
        x_positions = np.arange(0.5, num_steps + 0.5)
        y_position = 0.5
        
        # Draw steps
        for i, (x, step) in enumerate(zip(x_positions, steps)):
            circle = plt.Circle((x, y_position), 0.2, color='#3b82f6', zorder=2)
            ax.add_patch(circle)
            ax.text(x, y_position, f"{i+1}", color='white', fontsize=12, ha='center', va='center', zorder=3)
            ax.text(x, y_position - 0.3, step, fontsize=10, ha='center', va='top', rotation=45, color='#e2e8f0')
            
            # Draw arrows between steps
            if i < num_steps - 1:
                ax.arrow(x + 0.2, y_position, 0.6, 0, head_width=0.05, head_length=0.1, fc='#3b82f6', ec='#3b82f6', length_includes_head=True, zorder=1)
        
        plt.tight_layout()
        
        # Save the plot to a buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', transparent=True)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error in create_drug_discovery_pipeline_visualization: {str(e)}")
        raise