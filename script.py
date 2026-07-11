import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def generate_video(dimension, num_matrices, distribucion, fps, batch_size=1000):
    # ---- Generación de matrices y eigenvalues ----
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
    
    # ---- Límites de los ejes ----
    lim_real = max(1.0, np.percentile(np.abs(all_real), 99) * 1.1)
    lim_imag = max(1.0, np.percentile(np.abs(all_imag), 99) * 1.1)
    limite = max(lim_real, lim_imag)
    
    # ---- Generar frames (imágenes) ----
    num_frames = min(300, max(10, total_puntos // 500))
    step = max(1, total_puntos // num_frames)
    
    frames = []
    for frame_idx in range(num_frames):
        idx = min(total_puntos, (frame_idx + 1) * step)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-limite, limite)
        ax.set_ylim(-limite, limite)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel("Parte Real")
        ax.set_ylabel("Parte Imaginaria")
        ax.set_title(f"Eigenvalues mostrados: {idx} (de {total_puntos})")
        
        # Dibujar puntos acumulados
        ax.scatter(all_real[:idx], all_imag[:idx], s=1, alpha=0.6, color='#1f77b4')
        
        # Convertir a PNG en base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        frames.append(img_base64)
    
    # ---- Construir HTML con reproductor ----
    html = f'''
    <div id="eigen-player" style="text-align:center;">
        <img id="eigen-frame" src="data:image/png;base64,{frames[0]}" style="max-width:100%; border:1px solid #ddd; border-radius:4px;">
        <br>
        <button id="eigen-play" style="margin:10px 5px;">▶ Play</button>
        <button id="eigen-pause" style="margin:10px 5px;">⏸ Pause</button>
        <span id="eigen-counter">1 / {num_frames}</span>
        <br>
        <input type="range" id="eigen-slider" min="0" max="{num_frames-1}" value="0" style="width:80%;">
    </div>
    <script>
    (function() {{
        const frames = {frames};
        const total = frames.length;
        let current = 0;
        let intervalId = null;
        const img = document.getElementById('eigen-frame');
        const counter = document.getElementById('eigen-counter');
        const slider = document.getElementById('eigen-slider');
        const playBtn = document.getElementById('eigen-play');
        const pauseBtn = document.getElementById('eigen-pause');
        const fps = {fps};
        
        function showFrame(index) {{
            if (index >= 0 && index < total) {{
                img.src = 'data:image/png;base64,' + frames[index];
                current = index;
                counter.textContent = (index+1) + ' / ' + total;
                slider.value = index;
            }}
        }}
        
        function nextFrame() {{
            let next = (current + 1) % total;
            showFrame(next);
            if (next === 0) {{
                clearInterval(intervalId);
                intervalId = null;
                playBtn.textContent = '▶ Play';
            }}
        }}
        
        playBtn.addEventListener('click', function() {{
            if (intervalId === null) {{
                if (current === total - 1) showFrame(0);
                intervalId = setInterval(nextFrame, 1000/fps);
                playBtn.textContent = '⏸ Pause';
            }} else {{
                clearInterval(intervalId);
                intervalId = null;
                playBtn.textContent = '▶ Play';
            }}
        }});
        
        pauseBtn.addEventListener('click', function() {{
            if (intervalId !== null) {{
                clearInterval(intervalId);
                intervalId = null;
                playBtn.textContent = '▶ Play';
            }}
        }});
        
        slider.addEventListener('input', function() {{
            if (intervalId !== null) {{
                clearInterval(intervalId);
                intervalId = null;
                playBtn.textContent = '▶ Play';
            }}
            showFrame(parseInt(this.value));
        }});
        
        showFrame(0);
    }})();
    </script>
    '''
    return html