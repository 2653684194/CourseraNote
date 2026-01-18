import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_grid(ax, data, title, cmap='Blues', cell_size=1, offset=(0,0), show_values=True):
    rows, cols = data.shape
    ox, oy = offset
    
    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            # Normalize for color if needed
            # Use a fixed vmin/vmax for consistency if passed, otherwise auto
            color = plt.get_cmap(cmap)(val/data.max() if data.max() > 0 else 0)
            
            rect = patches.Rectangle((ox + j*cell_size, oy + (rows-1-i)*cell_size), cell_size, cell_size, 
                                   linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            if show_values:
                ax.text(ox + j*cell_size + cell_size/2, oy + (rows-1-i)*cell_size + cell_size/2, 
                        f'{int(val)}', ha='center', va='center', fontsize=10)
    
    ax.text(ox + cols*cell_size/2, oy + rows*cell_size + 0.2, title, ha='center', fontsize=12, fontweight='bold')

def generate_col2img_illustration():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 10)
    ax.axis('off')

    # Parameters
    H, W = 4, 4
    k = 2
    stride = 1
    h_out, w_out = 3, 3
    col_len = 4
    output_cols = 9
    
    # 1. Draw dZ_col Matrix on the Left
    # Shape 4x9
    dZ_col_data = np.ones((col_len, output_cols)) 
    # Just visuals, values don't matter as much as shape/color
    
    draw_grid(ax, dZ_col_data, "Gradient dZ_col (4x9)", offset=(0, 3), cmap='Greys', show_values=False)
    
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99'] 
    
    # Overlay colored rows
    for r in range(col_len):
        # In plot coords (bottom-up), row 0 is top
        # draw_grid puts row 0 at y = 3 + (4-1-0) = 6
        rect = patches.Rectangle((0, 3 + (col_len-1-r)), output_cols, 1, 
                               linewidth=2, edgecolor=colors[r], facecolor=colors[r], alpha=0.3)
        ax.add_patch(rect)
        ax.text(-1.5, 3 + (col_len-1-r) + 0.5, f"Row {r}\nFilter ({r//2},{r%2})", va='center', fontsize=9)

    # 2. Draw dX Matrix on the Right
    dX_data = np.zeros((H, W))
    draw_grid(ax, dX_data, "Gradient dX (4x4)", offset=(14, 3))
    
    # 3. Draw Arrows illustrating the mapping
    # Row 0 (Red) -> (0,0) shift
    ax.arrow(9.2, 6.5, 4.5, 0, head_width=0.2, color=colors[0], length_includes_head=True)
    
    # Row 3 (Yellow) -> (1,1) shift (bottom-right of 2x2)
    ax.arrow(9.2, 3.5, 4.5, -1, head_width=0.2, color=colors[3], length_includes_head=True)
    
    # Highlight Target Regions in dX
    # Region for Row 0 (3x3 top-left)
    rect0 = patches.Rectangle((14, 3+1), 3, 3, linewidth=2, edgecolor=colors[0], facecolor='none', linestyle='--')
    ax.add_patch(rect0)
    ax.text(15.5, 7.2, "Row 0 Adds Here", color=colors[0], ha='center', fontsize=9)
    
    # Region for Row 3 (3x3 bottom-right)
    # x=14+1=15, y=3+0=3
    rect3 = patches.Rectangle((15, 3), 3, 3, linewidth=2, edgecolor=colors[3], facecolor='none', linestyle=':')
    ax.add_patch(rect3)
    ax.text(16.5, 2.5, "Row 3 Adds Here", color='orange', ha='center', fontsize=9) # Use orange for visibility
    
    # Explanation Text
    text = (
        "Col2Img Process:\n"
        "1. Iterate through each ROW of dZ_col.\n"
        "2. Each row contains gradients for a specific filter pixel across the entire image.\n"
        "3. Reshape the row (1x9) into a 2D grid (3x3).\n"
        "4. Add this grid to dX, shifted by the filter position.\n"
        "   - Row 0 (Red): Shift (0,0)\n"
        "   - Row 3 (Yellow): Shift (1,1)"
    )
    ax.text(5, 0, text, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('d:/AAA_Jupyter/Coursera_ML/Ex10_CNN/layer_viz/col2img_illustration.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_col2img_illustration()
