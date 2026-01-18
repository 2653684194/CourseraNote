import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_grid(ax, data, title, cmap='Blues', cell_size=1, offset=(0,0)):
    rows, cols = data.shape
    ox, oy = offset
    
    for i in range(rows):
        for j in range(cols):
            rect = patches.Rectangle((ox + j*cell_size, oy + (rows-1-i)*cell_size), cell_size, cell_size, 
                                   linewidth=1, edgecolor='black', facecolor=plt.get_cmap(cmap)(data[i, j]))
            ax.add_patch(rect)
            ax.text(ox + j*cell_size + cell_size/2, oy + (rows-1-i)*cell_size + cell_size/2, 
                    f'{int(data[i, j])}', ha='center', va='center', fontsize=12)
    
    ax.text(ox + cols*cell_size/2, oy + rows*cell_size + 0.2, title, ha='center', fontsize=14, fontweight='bold')
    return ox + cols*cell_size, oy + rows*cell_size

def generate_img2col_viz():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-1, 18)
    ax.set_ylim(-1, 10)
    ax.axis('off')

    # Input Data 4x4
    input_data = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1]
    ]) * 0.2  # Scaling for color intensity
    
    # Kernel Size 2x2, Stride 1
    k = 2
    
    # Draw Input Image
    draw_grid(ax, input_data/input_data.max(), "Input Image (4x4)", offset=(0, 4))
    
    # Highlight Patch 1 (0,0)
    rect1 = patches.Rectangle((0, 4 + 2), 2, 2, linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect1)
    ax.text(1, 4 + 4.2, "Patch 1", color='red', ha='center')
    
    # Highlight Patch 2 (0,1)
    rect2 = patches.Rectangle((1, 4 + 2), 2, 2, linewidth=3, edgecolor='green', facecolor='none', linestyle='--')
    ax.add_patch(rect2)
    ax.text(2.5, 4 - 0.5, "Patch 2 (Stride 1)", color='green', ha='center')

    # Draw Arrow
    ax.arrow(5, 6, 2, 0, head_width=0.3, head_length=0.3, fc='k', ec='k')
    ax.text(6, 6.3, "Im2Col", ha='center')

    # Output Matrix (Column Matrix)
    # Each column is a flattened patch
    # Patch 1: [1, 2, 0, 1]
    # Patch 2: [2, 3, 1, 2]
    # ...
    
    # Let's show first 3 columns
    col_data = np.array([
        [1, 2, 3], # R1 (row 1 of patch)
        [2, 3, 0], # R1
        [0, 1, 2], # R2
        [1, 2, 3]  # R2
    ])
    
    # Normalize for color
    draw_grid(ax, col_data/3.0, "Output Matrix (X_col)", offset=(8, 4), cell_size=1)
    
    # Highlight Column 1
    rect_col1 = patches.Rectangle((8, 4), 1, 4, linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect_col1)
    
    # Highlight Column 2
    rect_col2 = patches.Rectangle((9, 4), 1, 4, linewidth=3, edgecolor='green', facecolor='none', linestyle='--')
    ax.add_patch(rect_col2)
    
    # Connectors
    # Input Patch 1 to Col 1
    # ax.plot([2, 8], [6, 8], 'r-', alpha=0.3)
    
    # Text Explanations
    text_x = 13
    text_y = 8
    line_height = 0.6
    
    ax.text(text_x, text_y, "Process:", fontweight='bold', fontsize=12)
    ax.text(text_x, text_y - line_height, "1. Sliding Window (Red Box)", fontsize=10)
    ax.text(text_x, text_y - 2*line_height, "   extracts 2x2 patch.", fontsize=10)
    ax.text(text_x, text_y - 3*line_height, "2. Flatten Patch -> Column", fontsize=10)
    ax.text(text_x, text_y - 4*line_height, "   [[1,2],[0,1]] -> [1,2,0,1]^T", fontsize=10)
    ax.text(text_x, text_y - 5*line_height, "3. Repeat for next stride (Green)", fontsize=10)
    ax.text(text_x, text_y - 6*line_height, "   Stack as new column.", fontsize=10)
    
    ax.text(text_x, text_y - 8*line_height, "Dimensions:", fontweight='bold', fontsize=12)
    ax.text(text_x, text_y - 9*line_height, "Input: (C, H, W)", fontsize=10)
    ax.text(text_x, text_y - 10*line_height, "Matrix: (C*f*f, H_out*W_out)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('d:/AAA_Jupyter/Coursera_ML/Ex10_CNN/img2col_illustration.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_img2col_viz()
