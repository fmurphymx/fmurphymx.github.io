import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_video(dimension, num_matrices, distribucion, fps, batch_size=1000):
    # ---- Generación de matrices ----
    all_real = []
    all_imag = []
    total_batches = int(np.ceil(num_matrices / batch_size))
    
    for i in range(total_batches):
        curr_batch = min(batch_size, num_matrices - i * batch_size)
        if distribucion == "Normal (Gaussiana)":
            matrices = np.random.randn(curr_batch, dimension, dimension)
        else:  # Uniforme
            matrices = np.random.uniform(-1, 1, (curr_batch, dimension, dimension))
        eigvals = np.linalg.eigvals(matrices)
        all_real.extend(eigvals.real.flatten())
        all_imag.extend(eigvals.imag.flatten())
    
    all_real = np.array(all_real)
    all_imag = np.array(all_imag)
    total_puntos = len(all_real)
    
    # ---- Configurar la figura ----
    fig, ax = plt.subplots(figsize=(6, 6))
    lim_real = max(1.0, np.percentile(np.abs(all_real), 99) * 1.1)
    lim_imag = max(1.0, np.percentile(np.abs(all_imag), 99) * 1.1)
    limite = max(lim_real, lim_imag)
    
    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)
    ax.set_aspect('equal')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel("Parte Real")
    ax.set_ylabel("Parte Imaginaria")
    
    scatter = ax.scatter([], [], s=1, alpha=0.6, color='#1f77b4')
    
    num_frames = min(300, max(10, total_puntos // 500))
    paso = max(1, total_puntos // num_frames)
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return scatter,
    
    def update(frame):
        idx = min(total_puntos, (frame + 1) * paso)
        datos = np.column_stack((all_real[:idx], all_imag[:idx]))
        scatter.set_offsets(datos)
        ax.set_title(f"Eigenvalues mostrados: {idx} (de {total_puntos})")
        return scatter,
    
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                        blit=True, repeat=False, interval=1000/fps)
    
    # Convertir a HTML5 (video)
    return ani.to_jshtml()