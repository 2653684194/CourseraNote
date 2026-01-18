import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

def create_img2col_animation():
    # Input Data 4x4
    input_data = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1]
    ])
    
    # Kernel 2x2, Stride 1
    k = 2
    stride = 1
    
    H, W = input_data.shape
    h_out = (H - k) // stride + 1
    w_out = (W - k) // stride + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.3)
    
    # Setup Input Heatmap (Grayscale for values)
    # We will overlay colored rectangles for the kernel
    im1 = ax1.imshow(input_data, cmap='Greys', vmin=0, vmax=5, alpha=0.3)
    ax1.set_title("Input Image (4x4)\nSliding Window (2x2)")
    ax1.set_xticks(np.arange(W))
    ax1.set_yticks(np.arange(H))
    ax1.grid(which='major', color='gray', linestyle=':', linewidth=0.5)
    
    # Add text to input
    for i in range(H):
        for j in range(W):
            ax1.text(j, i, f'{int(input_data[i, j])}', ha='center', va='center', fontsize=12, fontweight='bold')
            
    # Kernel Colors (Red, Green, Blue, Yellow) for (0,0), (0,1), (1,0), (1,1)
    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99'] # Light Red, Green, Blue, Yellow
    kernel_patches = []
    
    # Initialize kernel patches at top-left (hidden initially or at 0,0)
    for i in range(k):
        for j in range(k):
            # i is row (y), j is col (x)
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, 
                                   edgecolor=colors[i*k + j], facecolor=colors[i*k + j], alpha=0.5)
            ax1.add_patch(rect)
            kernel_patches.append(rect)
    
    # Setup Output Matrix (initially zeros)
    output_cols = h_out * w_out
    col_len = k * k
    output_data = np.zeros((col_len, output_cols))
    
    # Output heatmap
    im2 = ax2.imshow(output_data, cmap='Greys', vmin=0, vmax=5, alpha=0) # Invisible initially, we draw patches
    ax2.set_title(f"X_col Matrix ({col_len} rows x {output_cols} cols)\nEach column = Flattened Patch")
    ax2.set_xlabel("Patch Index (Batch Dimension)")
    ax2.set_ylabel("Feature Dimension (Flattened Kernel)")
    ax2.set_xticks(np.arange(output_cols))
    ax2.set_yticks(np.arange(col_len))
    ax2.set_yticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)']) # Explicit labels
    
    # Draw background stripes for output matrix to match kernel colors
    for r in range(col_len):
        # Draw a rectangle across the whole row
        rect = patches.Rectangle((-0.5, r - 0.5), output_cols, 1, 
                               linewidth=0, facecolor=colors[r], alpha=0.2)
        ax2.add_patch(rect)

    # Highlight current column
    rect_col = patches.Rectangle((-0.5, -0.5), 1, col_len, linewidth=3, edgecolor='black', facecolor='none')
    ax2.add_patch(rect_col)
    
    # Animation update function
    def update(frame):
        # Calculate current patch position
        idx_h = frame // w_out
        idx_w = frame % w_out
        
        start_h = idx_h * stride
        start_w = idx_w * stride
        
        # Update Window Position on Input
        for i in range(k):
            for j in range(k):
                p_idx = i*k + j
                # Move patch to (start_w + j, start_h + i)
                # Rectangle xy is bottom-left corner.
                # Grid coordinates (x, y) are centered at integers.
                # So pixel (x,y) extends from x-0.5 to x+0.5
                kernel_patches[p_idx].set_xy((start_w + j - 0.5, start_h + i - 0.5))
        
        # Update Output Data (Logic)
        current_output = np.zeros((col_len, output_cols))
        # Fill previous columns
        for f in range(frame + 1):
            h = f // w_out
            w = f % w_out
            sh = h * stride
            sw = w * stride
            p = input_data[sh:sh+k, sw:sw+k].flatten()
            current_output[:, f] = p
            
        # Draw values on Output
        # Remove existing texts manually
        for txt in ax2.texts:
            txt.remove()
        
        for i in range(col_len):
            for j in range(output_cols):
                if j <= frame:
                    val = current_output[i, j]
                    # Make current column text bold
                    weight = 'bold' if j == frame else 'normal'
                    # size = 12 if j == frame else 10
                    ax2.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=10, fontweight=weight)
        
        # Update Column Highlight
        rect_col.set_xy((frame - 0.5, -0.5))
        
        return kernel_patches + [rect_col] + ax2.texts + ax2.patches

    anim = FuncAnimation(fig, update, frames=h_out*w_out, interval=1000, blit=False)
    anim.save('d:/AAA_Jupyter/Coursera_ML/Ex10_CNN/layer_viz/img2col_animation.gif', writer='pillow', fps=1)
    plt.close()

if __name__ == "__main__":
    create_img2col_animation()
