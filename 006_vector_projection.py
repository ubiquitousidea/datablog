import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from matplotlib.patches import Polygon

def create_2d_projection_plot():
    # Setup figure
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 2D Geometry
    # O is origin (0, 0)
    # y_hat is (4, 2)
    # e is (-1, 2)
    # Y is y_hat + e = (3, 4)
    O = np.array([0, 0])
    yhat = np.array([4.0, 2.0])
    e = np.array([-1.0, 2.0])
    Y = yhat + e  # (3, 4)
    
    # Column space line direction (same as yhat)
    # Let's draw the column space as a line extending from -1 to 5.5
    cs_start = -0.3 * yhat
    cs_end = 1.3 * yhat
    ax.plot([cs_start[0], cs_end[0]], [cs_start[1], cs_end[1]], 
            color='#d1d5db', linestyle='--', linewidth=2, label='Column Space Col(X)')
    
    # Shade a region representing the column space plane (1D line in 2D)
    # To give a feel of "space", we can draw a subtle light blue band along the line
    # or just keep it simple and clean.
    
    # Draw right-angle marker
    # Box size
    d = 0.2
    # Unit vectors
    u_yhat = -yhat / np.linalg.norm(yhat)
    u_e = e / np.linalg.norm(e)
    
    p1 = yhat + d * u_yhat
    p3 = yhat + d * u_e
    p2 = p1 + d * u_e
    
    ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color='#9ca3af', linewidth=1.5)
    
    # Draw vectors using ax.annotate to get beautiful arrows
    # Vector y_hat (Blue)
    ax.annotate('', xy=yhat, xytext=O,
                arrowprops=dict(arrowstyle="->", color='#2563eb', lw=3, mutation_scale=20, shrinkA=0, shrinkB=0))
    # Vector Y (Indigo)
    ax.annotate('', xy=Y, xytext=O,
                arrowprops=dict(arrowstyle="->", color='#4f46e5', lw=3, mutation_scale=20, shrinkA=0, shrinkB=0))
    # Vector e (Orange-Red)
    ax.annotate('', xy=Y, xytext=yhat,
                arrowprops=dict(arrowstyle="->", color='#ea580c', lw=3, mutation_scale=20, shrinkA=0, shrinkB=0))
    
    # Add text labels
    # Label Y
    ax.text(Y[0]/2 - 0.3, Y[1]/2 + 0.3, r'$Y$', color='#4f46e5', fontsize=15, fontweight='bold')
    # Label y_hat
    ax.text(yhat[0]/2 + 0.2, yhat[1]/2 - 0.3, r'$\hat{y} = \beta X$', color='#2563eb', fontsize=15, fontweight='bold')
    # Label e
    ax.text(yhat[0] + e[0]/2 + 0.25, yhat[1] + e[1]/2, r'$e$', color='#ea580c', fontsize=15, fontweight='bold')
    
    # Label for Column Space
    ax.text(cs_end[0] - 0.8, cs_end[1] - 0.4, r'$\text{Col}(X)$', color='#9ca3af', fontsize=13, style='italic')
    
    # Adjust plot limits and clean axes
    ax.set_xlim(-1.5, 6.0)
    ax.set_ylim(-1.0, 5.0)
    ax.set_aspect('equal')
    
    # Light gray axes lines passing through origin
    ax.axhline(0, color='#e5e7eb', linewidth=1.2, zorder=-1)
    ax.axvline(0, color='#e5e7eb', linewidth=1.2, zorder=-1)
    
    # Hide standard spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    output_path = '006_projection_2d.png'
    plt.savefig(output_path, facecolor='white', dpi=150)
    plt.close()
    print(f"Saved 2D projection plot to {output_path}")

def create_3d_projection_plot():
    # Draw a 3D perspective diagram of projection onto a plane
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0],ys[0]), (xs[1],ys[1]))
            super().draw(renderer)
            
        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0],ys[0]), (xs[1],ys[1]))
            return np.min(zs)

    fig = plt.figure(figsize=(8, 7), dpi=150)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    # Geometry
    O = np.array([0, 0, 0])
    yhat = np.array([3.0, 2.5, 0.0])
    e = np.array([0.0, 0.0, 2.2])
    Y = yhat + e  # (3.0, 2.5, 2.2)
    
    # Columns of X to show they span the plane
    x1 = np.array([2.5, 0.5, 0.0])
    x2 = np.array([1.0, 2.2, 0.0])
    
    # Draw the column space plane as a polygon in the z=0 plane
    # Vertices of the plane patch
    plane_verts = [
        [-1.0, -1.0, 0.0],
        [5.0, -0.5, 0.0],
        [4.5, 4.0, 0.0],
        [-1.5, 3.5, 0.0]
    ]
    poly = Poly3DCollection([plane_verts], facecolors='#eff6ff', edgecolors='#93c5fd', alpha=0.6, linewidths=1.5)
    ax.add_collection3d(poly)
    
    # Draw right-angle marker in 3D
    # The right angle is between e (which is vertical along z) and the plane (z=0)
    # We can draw a small 3D corner box at yhat
    d = 0.25
    ax.plot([yhat[0] - d, yhat[0] - d, yhat[0]], 
            [yhat[1], yhat[1], yhat[1]], 
            [yhat[2], yhat[2] + d, yhat[2] + d], 
            color='#9ca3af', linewidth=1.2)
    ax.plot([yhat[0], yhat[0], yhat[0]], 
            [yhat[1] - d, yhat[1] - d, yhat[1]], 
            [yhat[2], yhat[2] + d, yhat[2] + d], 
            color='#9ca3af', linewidth=1.2)
    
    # Add vectors
    # Y (Indigo)
    arrow_Y = Arrow3D([O[0], Y[0]], [O[1], Y[1]], [O[2], Y[2]], 
                      mutation_scale=15, lw=3, arrowstyle="-|>", color='#4f46e5')
    ax.add_artist(arrow_Y)
    
    # yhat (Blue)
    arrow_yhat = Arrow3D([O[0], yhat[0]], [O[1], yhat[1]], [O[2], yhat[2]], 
                         mutation_scale=15, lw=3, arrowstyle="-|>", color='#2563eb')
    ax.add_artist(arrow_yhat)
    
    # e (Orange-Red)
    arrow_e = Arrow3D([yhat[0], Y[0]], [yhat[1], Y[1]], [yhat[2], Y[2]], 
                      mutation_scale=15, lw=3, arrowstyle="-|>", color='#ea580c')
    ax.add_artist(arrow_e)
    
    # Subtle column vectors X1 and X2 (Gray dashed or solid)
    arrow_x1 = Arrow3D([O[0], x1[0]], [O[1], x1[1]], [O[2], x1[2]], 
                       mutation_scale=10, lw=1.5, arrowstyle="-|>", color='#9ca3af', linestyle='--')
    arrow_x2 = Arrow3D([O[0], x2[0]], [O[1], x2[1]], [O[2], x2[2]], 
                       mutation_scale=10, lw=1.5, arrowstyle="-|>", color='#9ca3af', linestyle='--')
    ax.add_artist(arrow_x1)
    ax.add_artist(arrow_x2)
    
    # Labels
    # We use ax.text for labels, positioning them carefully
    ax.text(Y[0]/2 - 0.2, Y[1]/2 - 0.2, Y[2]/2 + 0.3, r'$Y$', color='#4f46e5', fontsize=15, fontweight='bold')
    ax.text(yhat[0]/2 + 0.1, yhat[1]/2 + 0.2, 0.05, r'$\hat{y} = \beta X$', color='#2563eb', fontsize=15, fontweight='bold')
    ax.text(yhat[0] + 0.1, yhat[1] + 0.1, Y[2]/2, r'$e$', color='#ea580c', fontsize=15, fontweight='bold')
    
    # Labels for columns of X
    ax.text(x1[0], x1[1] - 0.3, 0.0, r'$X_1$', color='#6b7280', fontsize=11)
    ax.text(x2[0] - 0.3, x2[1] + 0.1, 0.0, r'$X_2$', color='#6b7280', fontsize=11)
    
    # Label for column space
    ax.text(4.2, 3.8, 0.0, r'$\text{Col}(X)$', color='#3b82f6', fontsize=14, style='italic', fontweight='bold')
    
    # View & styling
    ax.set_xlim(-1.5, 5.5)
    ax.set_ylim(-1.5, 4.5)
    ax.set_zlim(-0.2, 3.0)
    
    # Remove grid, background, ticks
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # Hide axis lines and ticks
    ax.axis('off')
    
    # Set view angle to look good
    ax.view_init(elev=22, azim=-55)
    
    output_path = '006_projection_3d.png'
    plt.savefig(output_path, facecolor='white', dpi=150)
    plt.close()
    print(f"Saved 3D projection plot to {output_path}")

if __name__ == '__main__':
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    create_2d_projection_plot()
    create_3d_projection_plot()
    
    # Copy files to artifacts directory
    artifact_dir = os.environ.get('ANTIGRAVITY_ARTIFACT_DIR', '')
    if artifact_dir and os.path.exists(artifact_dir):
        shutil.copy('006_projection_2d.png', os.path.join(artifact_dir, '006_projection_2d.png'))
        shutil.copy('006_projection_3d.png', os.path.join(artifact_dir, '006_projection_3d.png'))
        print("Copied projection plots to artifacts directory.")
