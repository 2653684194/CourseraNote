import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

def create_col2img_animation():
    # Parameters
    H, W = 4, 4
    k = 2
    stride = 1
    h_out = (H - k) // stride + 1 # 3
    w_out = (W - k) // stride + 1 # 3
    output_cols = h_out * w_out # 9
    col_len = k * k # 4
    
    # Create distinct values for dZ_col to trace them
    # Shape (4, 9)
    # We use simple integers 1..36
    dZ_col_data = np.arange(1, col_len * output_cols + 1).reshape(col_len, output_cols)
    
    # Target dX
    dX = np.zeros((H, W))
    
    # Colors for kernel positions (rows of dZ_col)
    # Matching img2col: Red, Green, Blue, Yellow
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99'] 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.3)
    
    # Setup ax1: dZ_col Matrix
    im1 = ax1.imshow(dZ_col_data, cmap='Greys', vmin=0, vmax=50, alpha=0.1)
    ax1.set_title(f"Gradient dZ_col ({col_len}x{output_cols})\nRows = Filter Positions")
    ax1.set_ylabel("Filter Pos (Flattened)")
    ax1.set_xlabel("Patch Index")
    ax1.set_yticks(np.arange(col_len))
    ax1.set_xticks(np.arange(output_cols))
    ax1.set_yticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
    
    # Color code rows of dZ_col
    for r in range(col_len):
        rect = patches.Rectangle((-0.5, r - 0.5), output_cols, 1, 
                               linewidth=0, facecolor=colors[r], alpha=0.3)
        ax1.add_patch(rect)
        
    # Text for dZ_col
    texts_dZ = []
    for i in range(col_len):
        row_texts = []
        for j in range(output_cols):
            t = ax1.text(j, i, f'{int(dZ_col_data[i, j])}', ha='center', va='center', fontsize=9)
            row_texts.append(t)
        texts_dZ.append(row_texts)

    # Setup ax2: dX Accumulation
    im2 = ax2.imshow(dX, cmap='Blues', vmin=0, vmax=100)
    ax2.set_title("Gradient dX (Accumulation)\nSumming gradients back to pixels")
    ax2.set_xticks(np.arange(W))
    ax2.set_yticks(np.arange(H))
    ax2.grid(which='major', color='gray', linestyle=':', linewidth=0.5)
    
    # Text for dX values (sum)
    texts_dX = [[ax2.text(j, i, '0', ha='center', va='center', fontweight='bold') for j in range(W)] for i in range(H)]
    
    # Animation frames: loop over rows of dZ_col (filter positions)
    # We add a few frames at the start/end
    
    def update(frame):
        # Frame 0-3: Process Row 0-3
        row_idx = frame
        
        # Reset dX for visualization re-calculation
        current_dX = np.zeros((H, W))
        
        # Highlight current row in ax1
        [p.remove() for p in ax1.patches if isinstance(p, patches.Rectangle) and p.get_linewidth()==2]
        
        # Draw highlight box for current row
        rect_row = patches.Rectangle((-0.5, row_idx - 0.5), output_cols, 1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect_row)
        
        # Calculate dX accumulation up to and including this row
        for r in range(row_idx + 1):
            # Extract row data
            row_data = dZ_col_data[r, :]
            
            # Reshape 1D row (9,) -> 2D (3,3)
            # This corresponds to the gradients for that specific filter pixel
            # across the whole spatial map
            grad_slice = row_data.reshape(h_out, w_out)
            
            # Determine placement in dX
            fh, fw = r // k, r % k
            
            # Accumulate
            # dX slice: [fh : fh + h_out, fw : fw + w_out]
            current_dX[fh:fh+h_out, fw:fw+w_out] += grad_slice
            
        # Update dX display
        im2.set_data(current_dX)
        im2.set_clim(vmin=0, vmax=current_dX.max())
        
        # Update dX texts
        for i in range(H):
            for j in range(W):
                val = current_dX[i, j]
                texts_dX[i][j].set_text(f'{int(val)}')
        
        # Highlight the region in dX being updated THIS frame
        [p.remove() for p in ax2.patches]
        
        curr_fh, curr_fw = row_idx // k, row_idx % k
        rect_region = patches.Rectangle((curr_fw - 0.5, curr_fh - 0.5), w_out, h_out, 
                                      linewidth=2, edgecolor=colors[row_idx], facecolor=colors[row_idx], alpha=0.3, linestyle='--')
        ax2.add_patch(rect_region)
        ax2.set_xlabel(f"Adding Row {row_idx} (Filter {curr_fh},{curr_fw}) to region")

        return [im2, rect_row, rect_region] + [t for row in texts_dX for t in row]

    anim = FuncAnimation(fig, update, frames=col_len, interval=1500, blit=False)
    anim.save('d:/AAA_Jupyter/Coursera_ML/Ex10_CNN/layer_viz/col2img_animation.gif', writer='pillow', fps=0.75)
    plt.close()

if __name__ == "__main__":
    create_col2img_animation()
