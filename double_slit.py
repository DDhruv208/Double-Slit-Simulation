import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------
# Grid setup
# ---------------------------
Nx, Ny = 200, 200
Lx, Ly = 30.0, 30.0
dx, dy = Lx/Nx, Ly/Ny
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y, indexing='xy')

dt = 0.01
tmax = 5.0
steps = int(tmax/dt)  # 500 frames

# ---------------------------
# Initial Gaussian wavepacket
# ---------------------------
x0, y0 = 0, -10
sigma = 1.0
k0y = 5.0
psi = np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2)) * np.exp(1j*k0y*Y)
psi = psi.astype(np.complex128)

# ---------------------------
# Double-slit potential
# ---------------------------
V = np.zeros_like(X)
barrier_mask = (np.abs(Y) < 0.3)
V[barrier_mask] = 1e6

slit_half_width = 0.6
slit_offset = 3.0
for xc in [-slit_offset/2, +slit_offset/2]:
    slit_mask = barrier_mask & (np.abs(X-xc) < slit_half_width)
    V[slit_mask] = 0.0

# ---------------------------
# Operators
# ---------------------------
kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='xy')
T_prop = np.exp(-1j*0.5*(KX**2+KY**2)*dt)
V_half = np.exp(-1j*V*dt/2)

# ---------------------------
# Plot setup (cinematic look)
# ---------------------------
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(6, 6))
extent = [x.min(), x.max(), y.min(), y.max()]

im = ax.imshow(np.abs(psi)**2, extent=extent, origin='lower',
               cmap='inferno', vmin=0, vmax=0.05, interpolation="bicubic")

# Solid white barrier line at y=0
ax.axhline(y=0, color='white', linewidth=3)

# Create transparent gaps for slits (cover barrier with black)
for xc in [-slit_offset/2, +slit_offset/2]:
    ax.add_patch(plt.Rectangle(
        (xc-slit_half_width, -0.05),   # position (slightly thicker than line)
        2*slit_half_width, 0.1,        # width, height
        color='black', zorder=20
    ))


ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title("Double-Slit Interference", fontsize=14)

# Time counter
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                    color="white", fontsize=12)

# ---------------------------
# Time evolution function
# ---------------------------
def evolve(frame):
    global psi
    psi = V_half * psi
    psi_k = np.fft.fft2(psi)
    psi_k *= T_prop
    psi = np.fft.ifft2(psi_k)
    psi = V_half * psi

    # Update density plot
    density = np.abs(psi)**2
    im.set_data(density)
    im.set_clim(vmin=0, vmax=np.max(density)*0.5)  # auto-contrast

    time_text.set_text(f"t = {frame*dt:.2f}")
    return [im, time_text]

ani = FuncAnimation(fig, evolve, frames=steps, interval=30, blit=True)

# ---------------------------
# Save or Show
# ---------------------------
# For presentation video:
ani.save("double_slit.mp4", writer="ffmpeg", dpi=150)

print("Saved video as double_slit_t0-5.mp4")
