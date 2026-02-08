"""
Quantum Physics Calculator - Equation-Enhanced Edition (FIXED)
All existing features preserved + Governing equations displayed for every quantum system
All backslash/syntax errors resolved
"""

import os
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ============================================
# PHYSICAL CONSTANTS (UNCHANGED)
# ============================================
HBAR = 1.054571817e-34    # J·s
M_E = 9.10938356e-31      # kg
EV_TO_J = 1.602176634e-19
BOHR_RADIUS = 5.29177210903e-11
RYDBERG = 13.605693122994  # eV

# ============================================
# QUANTUM ENGINE (UNCHANGED - Exact Solutions)
# ============================================

class QuantumEngine:
    """Unified solver for multiple quantum systems - unchanged"""
    
    @staticmethod
    def solve_tunneling(E, V, L, m):
        """Rectangular barrier tunneling"""
        if E <= 0: 
            return 0.0, None
        
        if E > V:
            k1 = np.sqrt(2 * m * E) / HBAR
            k2 = np.sqrt(2 * m * (E - V)) / HBAR
            T = 4 * k1 * k2 / ((k1 + k2)**2)
        else:
            k = np.sqrt(2 * m * E) / HBAR
            kappa = np.sqrt(2 * m * (V - E)) / HBAR
            if kappa * L > 100:
                T = 0.0
            else:
                sinh_term = np.sinh(kappa * L)
                denom = 1 + (V**2 * sinh_term**2) / (4 * E * (V - E))
                T = 1 / denom
        
        # Generate wavefunction (unchanged)
        x = np.linspace(-5e-9, L + 5e-9, 1000)
        psi = np.zeros_like(x, dtype=complex)
        potential = np.zeros_like(x)
        barrier_mask = (x >= 0) & (x <= L)
        potential[barrier_mask] = V
        
        if E < V:
            kappa = np.sqrt(2 * m * (V - E)) / HBAR
            k = np.sqrt(2 * m * E) / HBAR
            A = 1.0
            denom = (k**2 + kappa**2) * np.sinh(kappa * L) + 2j * k * kappa * np.cosh(kappa * L)
            B = ((k**2 - kappa**2) * np.sinh(kappa * L)) / denom
            F = (2j * k * kappa) / denom
            
            for i, xi in enumerate(x):
                if xi < 0:
                    psi[i] = A * np.exp(1j * k * xi) + B * np.exp(-1j * k * xi)
                elif xi <= L:
                    C = F * np.exp(1j * k * L) * np.cosh(kappa * L/2)
                    D = F * np.exp(1j * k * L) * np.sinh(kappa * L/2)
                    psi[i] = C * np.exp(kappa * xi) + D * np.exp(-kappa * xi)
                else:
                    psi[i] = F * np.exp(1j * k * xi)
        else:
            k = np.sqrt(2 * m * E) / HBAR
            k2 = np.sqrt(2 * m * (E - V)) / HBAR
            A = 1.0
            B = ((k - k2) / (k + k2)) * A
            F = (2 * k / (k + k2)) * A
            
            for i, xi in enumerate(x):
                if xi < 0:
                    psi[i] = A * np.exp(1j * k * xi) + B * np.exp(-1j * k * xi)
                elif xi <= L:
                    C = F * np.exp(1j * k2 * L/2)
                    psi[i] = C * np.exp(1j * k2 * xi)
                else:
                    psi[i] = F * np.exp(1j * k * xi)
        
        return min(max(T, 0.0), 1.0), (x, np.real(psi), np.abs(psi)**2, potential)
    
    @staticmethod
    def solve_gaussian_tunneling(m, E_ev, V_ev, L_nm, x0_nm=-3.0, k0=5e9, sigma_nm=0.2, t_max=2e-15, dt=1e-17):
        """Time-dependent Schrödinger equation solver - unchanged"""
        E = E_ev * EV_TO_J
        V0 = V_ev * EV_TO_J
        L = L_nm * 1e-9
        x0 = x0_nm * 1e-9
        sigma = sigma_nm * 1e-9
        
        x_min, x_max = -10e-9, 10e-9
        N = 2048
        x = np.linspace(x_min, x_max, N)
        dx = x[1] - x[0]
        
        V = np.zeros_like(x)
        barrier_mask = (x >= 0) & (x <= L)
        V[barrier_mask] = V0
        
        psi = (1/(2*np.pi*sigma**2)**0.25) * np.exp(-(x-x0)**2/(4*sigma**2)) * np.exp(1j * k0 * x)
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
        
        k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        exp_V_half = np.exp(-1j * V * dt / (2 * HBAR))
        exp_T = np.exp(-1j * HBAR * k**2 * dt / (2 * m))
        
        frames = []
        times = np.arange(0, t_max, dt)
        store_interval = max(1, len(times) // 50)
        
        for step, t in enumerate(times):
            psi = exp_V_half * psi
            psi_k = np.fft.fft(psi)
            psi_k = exp_T * psi_k
            psi = np.fft.ifft(psi_k)
            psi = exp_V_half * psi
            
            if step % store_interval == 0:
                prob = np.abs(psi)**2
                trans_mask = x > L
                refl_mask = x < -L
                T = np.sum(prob[trans_mask]) * dx
                R = np.sum(prob[refl_mask]) * dx
                frames.append((t, x.copy(), np.real(psi), prob.copy(), V.copy(), T, R))
        
        return frames
    
    @staticmethod
    def solve_2d_tunneling(E_ev, V_ev, L_nm, angle_deg, m):
        """2D tunneling with angular incidence - unchanged"""
        E = E_ev * EV_TO_J
        V0 = V_ev * EV_TO_J
        L = L_nm * 1e-9
        theta = np.radians(angle_deg)
        E_perp = E * np.cos(theta)**2
        
        if E_perp > V0:
            k1 = np.sqrt(2 * m * E) / HBAR
            k2 = np.sqrt(2 * m * (E_perp - V0)) / HBAR
            T = 4 * k1 * k2 * np.cos(theta) / ((k1 * np.cos(theta) + k2)**2)
        elif E_perp > 0:
            kappa = np.sqrt(2 * m * (V0 - E_perp)) / HBAR
            if kappa * L > 50:
                T = 0.0
            else:
                sinh_term = np.sinh(kappa * L)
                denom = 1 + (V0**2 * sinh_term**2) / (4 * E_perp * (V0 - E_perp))
                T = 1 / denom
        else:
            T = 0.0
        
        x = np.linspace(-5e-9, L + 5e-9, 200)
        y = np.linspace(-3e-9, 3e-9, 150)
        X, Y = np.meshgrid(x, y)
        V_2d = np.zeros_like(X)
        barrier_mask = (X >= 0) & (X <= L)
        V_2d[barrier_mask] = V0
        
        kx = np.sqrt(2 * m * E_perp) / HBAR if E_perp > 0 else 0
        ky = np.sqrt(2 * m * E * np.sin(theta)**2) / HBAR if E * np.sin(theta)**2 > 0 else 0
        
        psi_2d = np.zeros_like(X, dtype=complex)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                xi, yi = X[i,j], Y[i,j]
                if xi < 0:
                    psi_2d[i,j] = np.exp(1j * (kx * xi + ky * yi))
                elif xi <= L:
                    if E_perp < V0:
                        kappa = np.sqrt(2 * m * (V0 - E_perp)) / HBAR
                        psi_2d[i,j] = np.exp(-kappa * xi) * np.exp(1j * ky * yi)
                    else:
                        k2 = np.sqrt(2 * m * (E_perp - V0)) / HBAR
                        psi_2d[i,j] = np.exp(1j * (k2 * xi + ky * yi))
                else:
                    psi_2d[i,j] = np.exp(1j * (kx * xi + ky * yi))
        
        prob_2d = np.abs(psi_2d)**2
        return T, X, Y, prob_2d, V_2d, theta
    
    @staticmethod
    def solve_particle_in_box(n, L, m):
        """Particle in a box - unchanged"""
        if n < 1 or L <= 0: 
            return 0.0, None
        
        E_n = (n**2 * np.pi**2 * HBAR**2) / (2 * m * L**2)
        x = np.linspace(0, L, 1000)
        psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
        prob = psi**2
        potential = np.zeros_like(x)
        potential[0] = 1e-18
        
        return E_n / EV_TO_J, (x, psi, prob, potential, E_n)
    
    @staticmethod
    def solve_harmonic_oscillator(n, omega, m):
        """Harmonic oscillator - unchanged"""
        if n < 0: 
            return 0.0, None
        
        E_n = HBAR * omega * (n + 0.5)
        x = np.linspace(-5e-10, 5e-10, 1000)
        alpha = np.sqrt(m * omega / HBAR)
        xi = alpha * x
        
        if n == 0:
            Hn = np.ones_like(xi)
        elif n == 1:
            Hn = 2 * xi
        elif n == 2:
            Hn = 4 * xi**2 - 2
        else:
            Hn_minus2 = np.ones_like(xi)
            Hn_minus1 = 2 * xi
            for k in range(2, n+1):
                Hn = 2 * xi * Hn_minus1 - 2*(k-1) * Hn_minus2
                Hn_minus2, Hn_minus1 = Hn_minus1, Hn
        
        psi = (alpha/np.pi)**0.25 * np.exp(-xi**2/2) * Hn / np.sqrt(2**n * np.math.factorial(n))
        prob = psi**2
        potential = 0.5 * m * omega**2 * x**2
        
        return E_n / EV_TO_J, (x, psi, prob, potential, E_n)
    
    @staticmethod
    def solve_hydrogen_atom(n, l, m_qn):
        """Hydrogen atom - unchanged"""
        if n < 1 or l >= n or abs(m_qn) > l:
            return None, None
        
        E_n = -RYDBERG / n**2
        r = np.linspace(0, 5e-10, 1000)
        a0 = BOHR_RADIUS
        
        if n == 1 and l == 0:
            R = 2 * (1/a0)**1.5 * np.exp(-r/a0)
        elif n == 2 and l == 0:
            R = (1/np.sqrt(2)) * (1/a0)**1.5 * (1 - r/(2*a0)) * np.exp(-r/(2*a0))
        elif n == 2 and l == 1:
            R = (1/np.sqrt(24)) * (1/a0)**1.5 * (r/a0) * np.exp(-r/(2*a0))
        else:
            R = np.exp(-r/(n*a0)) * (r/a0)**l
        
        prob = r**2 * R**2
        potential = -EV_TO_J * RYDBERG * (a0 / r)
        potential[0] = potential[1]
        
        return E_n, (r, R, prob, potential)

# ============================================
# EQUATION METADATA DICTIONARY (FIXED - Raw Strings for LaTeX)
# ============================================

def get_equation_metadata(system_type):
    """Returns governing equation, solution formula, interpretation and applications for each quantum system"""
    # FIXED: All LaTeX strings use raw strings (r"") to avoid backslash escape issues
    equations = {
        "Quantum Tunneling (Static)": {
            "title": "Time-Independent Schrödinger Equation",
            "equation": r"$$-\frac{\hbar^2}{2m}\frac{d^2\psi(x)}{dx^2} + V(x)\psi(x) = E\psi(x)$$",
            "solution": r"$$T = \left[1 + \frac{V_0^2\sinh^2(\kappa L)}{4E(V_0-E)}\right]^{-1}, \quad \kappa = \frac{\sqrt{2m(V_0-E)}}{\hbar}$$",
            "interpretation": r"Wavefunction decays exponentially inside barrier ($\psi \sim e^{-\kappa x}$), enabling classically forbidden penetration",
            "application": "Flash memory storage, Scanning Tunneling Microscopy (STM), Alpha decay in nuclei"
        },
        "Gaussian Wave Packet (Time-Dependent)": {
            "title": "Time-Dependent Schrödinger Equation",
            "equation": r"$$i\hbar\frac{\partial\psi(x,t)}{\partial t} = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi(x,t)$$",
            "solution": r"$$\psi(x,0) = \frac{1}{(2\pi\sigma^2)^{1/4}}e^{-(x-x_0)^2/4\sigma^2}e^{ik_0x}$$",
            "interpretation": r"Wave packet spreads due to dispersion ($\Delta x \Delta p \geq \hbar/2$); partial transmission/reflection creates interference patterns",
            "application": "Ultrafast electron microscopy, Quantum computing gate operations, Attosecond physics"
        },
        "2D Tunneling with Angle": {
            "title": "2D Schrödinger Equation with Angular Dependence",
            "equation": r"$$-\frac{\hbar^2}{2m}\nabla^2\psi + V(x)\psi = E\psi, \quad \nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$$",
            "solution": r"$$T(\theta) = \left[1 + \frac{V_0^2\sinh^2(\kappa L)}{4E\cos^2\theta\,(V_0-E\cos^2\theta)}\right]^{-1}, \quad \kappa = \frac{\sqrt{2m(V_0-E\cos^2\theta)}}{\hbar}$$",
            "interpretation": r"Only momentum component normal to barrier affects tunneling: $E_{\perp} = E\cos^2\theta$",
            "application": "STM tip geometry optimization, Angle-resolved photoemission spectroscopy (ARPES)"
        },
        "Particle in a Box": {
            "title": "Infinite Square Well Potential",
            "equation": r"$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi, \quad \psi(0)=\psi(L)=0$$",
            "solution": r"$$\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right), \quad E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$$",
            "interpretation": r"Boundary conditions quantize energy levels; wavefunction has $n-1$ nodes",
            "application": "Quantum dots, Nanowire electronics, Conjugated polymers in organic LEDs"
        },
        "Harmonic Oscillator": {
            "title": "Quantum Harmonic Oscillator",
            "equation": r"$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + \frac{1}{2}m\omega^2x^2\psi = E\psi$$",
            "solution": r"$$E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad \psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}\frac{1}{\sqrt{2^nn!}}H_n(\xi)e^{-\xi^2/2}$$",
            "interpretation": r"Zero-point energy ($\frac{1}{2}\hbar\omega$) prevents localization at equilibrium; equally spaced energy levels",
            "application": "Molecular vibrations, Quantum optics (photon states), Lattice dynamics in solids"
        },
        "Hydrogen Atom": {
            "title": "Hydrogen Atom (Coulomb Potential)",
            "equation": r"$$\left[-\frac{\hbar^2}{2\mu}\nabla^2 - \frac{e^2}{4\pi\epsilon_0 r}\right]\psi = E\psi$$",
            "solution": r"$$E_n = -\frac{13.6\,\text{eV}}{n^2}, \quad \psi_{nlm}(r,\theta,\phi) = R_{nl}(r)Y_{lm}(\theta,\phi)$$",
            "interpretation": r"Three quantum numbers ($n,l,m$) from spherical symmetry; degeneracy lifted by external fields",
            "application": "Atomic spectroscopy, Quantum chemistry, Laser physics, Stellar nucleosynthesis"
        }
    }
    return equations.get(system_type, equations["Quantum Tunneling (Static)"])

# ============================================
# VISUALIZATIONS (UNCHANGED - Plotly Fixed)
# ============================================

def create_tunneling_plot_dark(data, params):
    if data is None:
        fig = go.Figure()
        fig.add_annotation(text="No wavefunction data", xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False, 
                         font=dict(size=18, color="#e0e0e0"))
        fig.update_layout(height=550, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                         font=dict(color='#e0e0e0'))
        return fig
    
    x, psi_real, psi_prob, potential = data
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, row_heights=[0.65, 0.35])
    
    fig.add_trace(go.Scatter(x=x*1e9, y=psi_real, name='Wave Function', 
                            line=dict(color='#64b5f6', width=2.8),
                            fill='tozeroy', fillcolor='rgba(100, 181, 246, 0.12)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x*1e9, y=psi_prob, name='Probability', 
                            line=dict(color='#81c784', width=2.3),
                            fill='tozeroy', fillcolor='rgba(129, 199, 132, 0.1)'), row=1, col=1)
    
    L = params['barrier_width']
    fig.add_vrect(x0=0, x1=L, fillcolor="rgba(229, 115, 115, 0.1)", line_width=0, row=1, col=1)
    
    fig.add_trace(go.Scatter(x=x*1e9, y=potential/EV_TO_J, name='Potential', 
                            line=dict(color='#e57373', width=3),
                            fill='tozeroy', fillcolor='rgba(229, 115, 115, 0.06)'), row=2, col=1)
    
    fig.update_layout(
        height=550,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(family="Segoe UI, Arial", size=13, color='#e0e0e0'),
        showlegend=False,
        margin=dict(l=50, r=30, t=30, b=50),
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)', 
                    zeroline=True, zerolinecolor='rgba(255,255,255,0.15)', row=2, col=1,
                    title_text="Position (nm)", title_font_color='#bbbbbb')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)',
                    title_text="Amplitude", title_font_color='#bbbbbb', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)',
                    title_text="Energy (eV)", title_font_color='#bbbbbb', row=2, col=1)
    return fig

def create_gaussian_evolution_plot(frames, current_frame=0):
    if not frames or current_frame >= len(frames):
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False, 
                         font=dict(size=18, color="#e0e0e0"))
        fig.update_layout(height=580, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                         font=dict(color='#e0e0e0'))
        return fig
    
    t, x, psi_real, prob, V, T, R = frames[current_frame]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, row_heights=[0.65, 0.35])
    
    fig.add_trace(go.Scatter(x=x*1e9, y=psi_real, name='Wave Function (Re ψ)', 
                            line=dict(color='#64b5f6', width=2.5),
                            fill='tozeroy', fillcolor='rgba(100, 181, 246, 0.15)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x*1e9, y=prob, name='Probability Density |ψ|²', 
                            line=dict(color='#81c784', width=2.8),
                            fill='tozeroy', fillcolor='rgba(129, 199, 132, 0.2)'), row=1, col=1)
    
    L = 0.3
    fig.add_vrect(x0=0, x1=L, fillcolor="rgba(229, 115, 115, 0.15)", line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=x*1e9, y=V/EV_TO_J, name='Potential Barrier', 
                            line=dict(color='#e57373', width=3),
                            fill='tozeroy', fillcolor='rgba(229, 115, 115, 0.08)'), row=2, col=1)
    
    fig.update_layout(
        height=580,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(family="Segoe UI, Arial", size=13, color='#e0e0e0'),
        showlegend=True,
        legend=dict(font_color='#e0e0e0', yanchor="top", y=0.98, xanchor="left", x=0.01),
        margin=dict(l=50, r=30, t=40, b=60),
        hovermode='x unified',
        title=f"Time Evolution: t = {t*1e15:.2f} fs | Transmission: {T*100:.1f}% | Reflection: {R*100:.1f}%",
        title_font_color='#64b5f6',
        title_x=0.5
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)', 
                    zeroline=True, zerolinecolor='rgba(255,255,255,0.15)',
                    title_text="Position (nm)", title_font_color='#bbbbbb', row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", title_font_color='#bbbbbb', row=1, col=1)
    fig.update_yaxes(title_text="Energy (eV)", title_font_color='#bbbbbb', row=2, col=1,
                    range=[0, max(V/EV_TO_J)*1.1] if max(V)>0 else [0, 1])
    return fig

def create_2d_tunneling_plot(X, Y, prob_2d, V_2d, angle_deg):
    # FIXED: Removed invalid 'titleside' property
    fig = go.Figure(data=go.Contour(
        x=X[0]*1e9, 
        y=Y[:,0]*1e9, 
        z=prob_2d,
        colorscale='Viridis',
        contours=dict(showlabels=True, labelfont=dict(size=12, color='white')),
        colorbar=dict(
            title="Probability Density",
            tickfont=dict(color='#e0e0e0'),
            titlefont=dict(color='#e0e0e0')
        )
    ))
    
    L = 0.3
    fig.add_shape(type="rect", x0=0, y0=-3, x1=L, y1=3,
                 fillcolor="rgba(229, 115, 115, 0.3)", line_width=0)
    
    fig.add_annotation(
        x=-2, y=0.5,
        ax=-4, ay=0.5,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=3,
        arrowcolor="#64b5f6",
        text=f"{angle_deg}°",
        font=dict(color="#64b5f6", size=14)
    )
    
    fig.update_layout(
        height=550,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(family="Segoe UI", size=13, color='#e0e0e0'),
        xaxis_title="X Position (nm)", 
        yaxis_title="Y Position (nm)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', title_font_color='#bbbbbb'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', title_font_color='#bbbbbb'),
        title=f"2D Quantum Tunneling at {angle_deg}° Incidence",
        title_font_color='#64b5f6',
        title_x=0.5
    )
    return fig

# ============================================
# INTELLIGENT PARAMETER RANGES (UNCHANGED)
# ============================================

def get_particle_params(particle_type):
    particles = {
        "Electron": {
            "mass": M_E,
            "energy_range": (0.01, 50.0),
            "energy_default": 4.0,
            "barrier_height_range": (0.1, 60.0),
            "barrier_height_default": 5.0,
            "barrier_width_range": (0.01, 3.0),
            "barrier_width_default": 0.3,
            "box_width_range": (0.05, 20.0),
            "box_width_default": 1.0,
            "freq_range": (1, 200),
            "freq_default": 30,
            "gaussian_sigma_range": (0.05, 1.0),
            "gaussian_sigma_default": 0.2,
            "gaussian_k0_range": (1e9, 2e10),
            "gaussian_k0_default": 5e9,
            "angle_range": (0, 85),
            "angle_default": 30,
            "color": "#64b5f6",
            "symbol": "e⁻"
        },
        "Proton": {
            "mass": 1.6726e-27,
            "energy_range": (0.0001, 1.0),
            "energy_default": 0.01,
            "barrier_height_range": (0.001, 2.0),
            "barrier_height_default": 0.05,
            "barrier_width_range": (0.001, 0.5),
            "barrier_width_default": 0.05,
            "box_width_range": (0.001, 1.0),
            "box_width_default": 0.05,
            "freq_range": (0.1, 50),
            "freq_default": 5,
            "gaussian_sigma_range": (0.001, 0.05),
            "gaussian_sigma_default": 0.01,
            "gaussian_k0_range": (1e8, 5e9),
            "gaussian_k0_default": 1e9,
            "angle_range": (0, 85),
            "angle_default": 30,
            "color": "#e57373",
            "symbol": "p⁺"
        },
        "Neutron": {
            "mass": 1.6749e-27,
            "energy_range": (0.0001, 1.0),
            "energy_default": 0.01,
            "barrier_height_range": (0.001, 2.0),
            "barrier_height_default": 0.05,
            "barrier_width_range": (0.001, 0.5),
            "barrier_width_default": 0.05,
            "box_width_range": (0.001, 1.0),
            "box_width_default": 0.05,
            "freq_range": (0.1, 50),
            "freq_default": 5,
            "gaussian_sigma_range": (0.001, 0.05),
            "gaussian_sigma_default": 0.01,
            "gaussian_k0_range": (1e8, 5e9),
            "gaussian_k0_default": 1e9,
            "angle_range": (0, 85),
            "angle_default": 30,
            "color": "#81c784",
            "symbol": "n⁰"
        },
        "Alpha Particle": {
            "mass": 6.64e-27,
            "energy_range": (0.00001, 0.1),
            "energy_default": 0.001,
            "barrier_height_range": (0.0001, 0.5),
            "barrier_height_default": 0.01,
            "barrier_width_range": (0.0001, 0.1),
            "barrier_width_default": 0.01,
            "box_width_range": (0.0001, 0.2),
            "box_width_default": 0.01,
            "freq_range": (0.01, 10),
            "freq_default": 1,
            "gaussian_sigma_range": (0.0001, 0.01),
            "gaussian_sigma_default": 0.001,
            "gaussian_k0_range": (1e7, 1e9),
            "gaussian_k0_default": 2e8,
            "angle_range": (0, 85),
            "angle_default": 30,
            "color": "#ba68c8",
            "symbol": "α"
        },
        "Custom": {
            "mass": M_E,
            "energy_range": (0.00001, 100.0),
            "energy_default": 1.0,
            "barrier_height_range": (0.0001, 100.0),
            "barrier_height_default": 10.0,
            "barrier_width_range": (0.0001, 10.0),
            "barrier_width_default": 1.0,
            "box_width_range": (0.0001, 50.0),
            "box_width_default": 5.0,
            "freq_range": (0.01, 500),
            "freq_default": 50,
            "gaussian_sigma_range": (0.00001, 5.0),
            "gaussian_sigma_default": 0.5,
            "gaussian_k0_range": (1e6, 1e11),
            "gaussian_k0_default": 1e10,
            "angle_range": (0, 89),
            "angle_default": 45,
            "color": "#ffca28",
            "symbol": "?"
        }
    }
    return particles.get(particle_type, particles["Electron"])

# ============================================
# PROFESSIONAL DARK THEME UI WITH EQUATION DISPLAY (FIXED)
# ============================================

def create_professional_app():
    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: 0 auto !important; background: #0f0f1b !important; }
    .header { background: linear-gradient(135deg, #0f0f1b, #16213e, #1a1a2e); 
              padding: 2.2rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem; 
              box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 1px solid rgba(100, 181, 246, 0.2); }
    .sidebar { background: linear-gradient(135deg, #162447, #1a1a2e); padding: 1.8rem; border-radius: 18px; 
               box-shadow: 0 8px 25px rgba(0,0,0,0.4); margin-bottom: 1.8rem; border: 1px solid rgba(100, 181, 246, 0.15); }
    .main-content { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 2rem; border-radius: 18px; 
                    box-shadow: 0 8px 25px rgba(0,0,0,0.4); border: 1px solid rgba(100, 181, 246, 0.1); }
    .equation-panel { background: linear-gradient(135deg, #1a1a35, #121228); border-radius: 20px; padding: 28px; 
                      border: 2px solid rgba(138, 43, 226, 0.4); margin: 25px 0; box-shadow: 0 6px 20px rgba(0,0,0,0.35); }
    .calculate-btn { background: linear-gradient(135deg, #0f0f1b, #16213e); 
                     color: #64b5f6; border: 2px solid #64b5f6; border-radius: 14px; 
                     padding: 1.1rem 2.2rem; font-size: 1.25rem; font-weight: 700; width: 100%; 
                     box-shadow: 0 6px 22px rgba(100, 181, 246, 0.2); margin: 1.2rem 0; 
                     transition: all 0.3s; letter-spacing: 0.5px; }
    .calculate-btn:hover { background: linear-gradient(135deg, #16213e, #0f0f1b); 
                           transform: translateY(-3px); box-shadow: 0 9px 28px rgba(100, 181, 246, 0.35); 
                           border-color: #90caf9; }
    .particle-card { background: rgba(100, 181, 246, 0.08); border-radius: 14px; padding: 16px; 
                     border-left: 3px solid #64b5f6; margin-bottom: 12px; cursor: pointer; 
                     transition: all 0.3s; border: 1px solid rgba(100, 181, 246, 0.2); }
    .particle-card:hover { background: rgba(100, 181, 246, 0.15); transform: translateX(5px); 
                           border-color: rgba(100, 181, 246, 0.4); }
    .particle-card.selected { background: rgba(100, 181, 246, 0.2); border-left: 4px solid #90caf9; 
                              border-color: rgba(100, 181, 246, 0.5); }
    .quantum-badge { display: inline-block; background: rgba(100, 181, 246, 0.15); color: #64b5f6; 
                     padding: 4px 14px; border-radius: 18px; font-size: 0.95em; font-weight: 600; 
                     margin: 0 6px; letter-spacing: 0.5px; border: 1px solid rgba(100, 181, 246, 0.3); }
    .result-card { background: linear-gradient(135deg, rgba(100, 181, 246, 0.1), rgba(32, 60, 114, 0.2)); 
                   border-radius: 20px; padding: 35px 50px; text-align: center; 
                   box-shadow: 0 10px 30px rgba(32, 60, 114, 0.35); border: 1px solid rgba(100, 181, 246, 0.25); }
    .application-highlight { background: rgba(255, 215, 0, 0.12); border-radius: 14px; padding: 18px; 
                             border-left: 4px solid #ffc107; margin: 20px 0; }
    """
    
    with gr.Blocks(theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
    ).set(
        background_fill_primary="#0f0f1b",
        background_fill_secondary="#16213e",
        block_background_fill="#1a1a2e",
        block_border_color="#64b5f6",
        block_border_width="1px",
        button_primary_background_fill="#1e3c72",
        button_primary_background_fill_hover="#2a5298",
        button_primary_border_color="#64b5f6",
        button_primary_text_color="#64b5f6",
        body_text_color="#e0e0e0",
        body_text_color_subdued="#bbbbbb",
    ), css=custom_css) as app:
        
        # HEADER (UNCHANGED)
        gr.HTML("""
        <div class="header">
            <div style="font-size: 1.5rem; letter-spacing: 4px; opacity: 0.85; margin-bottom: 18px; font-weight: 300; text-transform: uppercase;">
                QUANTUM PHYSICS CALCULATOR
            </div>
            <h1 style="margin: 0; font-size: 3.3rem; font-weight: 800; letter-spacing: -1.2px; 
                       background: linear-gradient(to right, #64b5f6, #90caf9, #e3f2fd); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       text-shadow: 0 4px 25px rgba(100, 181, 246, 0.3);">
                Universal Quantum Solver
            </h1>
            <p style="margin: 22px 0 0 0; font-size: 1.5rem; opacity: 0.9; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.6;">
                Exact analytical solutions for <span class="quantum-badge">Tunneling</span> 
                <span class="quantum-badge">Particle in Box</span> 
                <span class="quantum-badge">Harmonic Oscillator</span> 
                <span class="quantum-badge">Hydrogen Atom</span>
            </p>
            <div style="margin-top: 25px; padding: 12px 30px; background: rgba(100, 181, 246, 0.1); 
                        display: inline-block; border-radius: 25px; font-size: 1.15rem; font-weight: 500; 
                        border: 1px solid rgba(100, 181, 246, 0.3);">
                <span style="margin-right: 25px;">🔬 Exact Schrödinger Solutions</span>
                <span style="margin-right: 25px;">⚡ Real-Time Computation</span>
                <span>🌌 Equation-Enhanced Learning</span>
            </div>
        </div>
        """)
        
        with gr.Row(equal_height=False):
            # SIDEBAR (UNCHANGED)
            with gr.Column(scale=1):
                gr.HTML('<div class="sidebar">')
                
                gr.Markdown("### 🧪 Particle Selection")
                selected_particle = gr.State(value="Electron")
                
                with gr.Row():
                    electron_card = gr.Button("🪫 Electron\n0.511 MeV/c²", elem_classes=["particle-card", "selected"])
                    proton_card = gr.Button("🟥 Proton\n938.3 MeV/c²", elem_classes="particle-card")
                with gr.Row():
                    neutron_card = gr.Button("🟢 Neutron\n939.6 MeV/c²", elem_classes="particle-card")
                    alpha_card = gr.Button("🟠 Alpha\n3.73 GeV/c²", elem_classes="particle-card")
                with gr.Row():
                    custom_card = gr.Button("⚙️ Custom Particle", elem_classes="particle-card")
                
                custom_mass_input = gr.Number(M_E, label="Custom Mass (kg)", visible=False)
                
                # Particle selection handler (unchanged)
                def update_particle_selection(particle_name):
                    updates = [
                        gr.update(elem_classes="particle-card selected" if particle_name == "Electron" else "particle-card"),
                        gr.update(elem_classes="particle-card selected" if particle_name == "Proton" else "particle-card"),
                        gr.update(elem_classes="particle-card selected" if particle_name == "Neutron" else "particle-card"),
                        gr.update(elem_classes="particle-card selected" if particle_name == "Alpha Particle" else "particle-card"),
                        gr.update(elem_classes="particle-card selected" if particle_name == "Custom" else "particle-card"),
                        gr.update(visible=(particle_name == "Custom")),
                        particle_name
                    ]
                    return tuple(updates)
                
                electron_card.click(lambda: update_particle_selection("Electron"), 
                                  outputs=[electron_card, proton_card, neutron_card, alpha_card, custom_card, custom_mass_input, selected_particle])
                proton_card.click(lambda: update_particle_selection("Proton"), 
                                outputs=[electron_card, proton_card, neutron_card, alpha_card, custom_card, custom_mass_input, selected_particle])
                neutron_card.click(lambda: update_particle_selection("Neutron"), 
                                 outputs=[electron_card, proton_card, neutron_card, alpha_card, custom_card, custom_mass_input, selected_particle])
                alpha_card.click(lambda: update_particle_selection("Alpha Particle"), 
                               outputs=[electron_card, proton_card, neutron_card, alpha_card, custom_card, custom_mass_input, selected_particle])
                custom_card.click(lambda: update_particle_selection("Custom"), 
                                outputs=[electron_card, proton_card, neutron_card, alpha_card, custom_card, custom_mass_input, selected_particle])
                
                # QUANTUM SYSTEM SELECTION (UNCHANGED)
                gr.Markdown("### ⚛️ Quantum System")
                equation_type = gr.Dropdown(
                    ["Quantum Tunneling (Static)", "Gaussian Wave Packet (Time-Dependent)", 
                     "2D Tunneling with Angle", "Particle in a Box", "Harmonic Oscillator", "Hydrogen Atom"],
                    value="Quantum Tunneling (Static)", label="Select Quantum Model"
                )
                
                # PRESETS (UNCHANGED)
                gr.Markdown("### 🚀 Quick Presets")
                with gr.Row():
                    with gr.Column():
                        btn_tunneling = gr.Button("✨ Electron Tunneling", elem_classes="demo-btn")
                        btn_box = gr.Button("📦 Quantum Dot", elem_classes="demo-btn")
                    with gr.Column():
                        btn_oscillator = gr.Button("🔬 Molecular Vibration", elem_classes="demo-btn")
                        btn_hydrogen = gr.Button("⚛️ Hydrogen 1s Orbital", elem_classes="demo-btn")
                
                gr.HTML('</div>')
            
            # MAIN CONTENT (ENHANCED WITH EQUATION PANEL)
            with gr.Column(scale=2):
                gr.HTML('<div class="main-content">')
                
                # === NEW: EQUATION DISPLAY PANEL ===
                gr.Markdown("### 📐 Governing Equation")
                equation_display = gr.Markdown("""
                <div class="equation-panel">
                    <h3 style="color: #64b5f6; margin-top: 0; text-align: center; font-size: 1.9rem; margin-bottom: 20px;">
                        Select a quantum system to view its governing equation
                    </h3>
                    <p style="text-align: center; color: #bbdefb; font-size: 1.25rem; max-width: 700px; margin: 0 auto;">
                        Every simulation displays the exact Schrödinger equation used for calculation, 
                        along with its analytical solution and physical interpretation.
                    </p>
                </div>
                """)
                
                # PARAMETERS (UNCHANGED)
                gr.Markdown("### 🎛️ System Parameters")
                
                with gr.Group(visible=True) as tunneling_group:
                    gr.Markdown("#### Barrier Parameters")
                    energy = gr.Slider(0.1, 20.0, 4.0, label="Particle Energy (eV)")
                    barrier_h = gr.Slider(0.5, 30.0, 5.0, label="Barrier Height (eV)")
                    barrier_w = gr.Slider(0.05, 2.0, 0.3, step=0.01, label="Barrier Width (nm)")
                
                with gr.Group(visible=False) as gaussian_group:
                    gr.Markdown("#### Gaussian Wave Packet Parameters")
                    gaussian_sigma = gr.Slider(0.05, 1.0, 0.2, step=0.01, label="Initial Width σ (nm)")
                    gaussian_k0 = gr.Slider(1e9, 2e10, 5e9, label="Initial Momentum k₀ (m⁻¹)")
                    gaussian_x0 = gr.Slider(-5.0, -1.0, -3.0, label="Initial Position x₀ (nm)")
                    gaussian_tmax = gr.Slider(0.5, 5.0, 2.0, label="Simulation Time (fs)")
                
                with gr.Group(visible=False) as angle_group:
                    gr.Markdown("#### Angular Tunneling Parameters")
                    angle_slider = gr.Slider(0, 85, 30, step=1, label="Incidence Angle θ (degrees)")
                    energy_2d = gr.Slider(0.1, 20.0, 4.0, label="Particle Energy (eV)")
                    barrier_h_2d = gr.Slider(0.5, 30.0, 5.0, label="Barrier Height (eV)")
                    barrier_w_2d = gr.Slider(0.05, 2.0, 0.3, step=0.01, label="Barrier Width (nm)")
                
                with gr.Group(visible=False) as box_group:
                    gr.Markdown("#### Box Parameters")
                    box_n = gr.Slider(1, 10, 1, step=1, label="Quantum Number n")
                    box_width = gr.Slider(0.1, 10.0, 1.0, label="Box Width (nm)")
                
                with gr.Group(visible=False) as oscillator_group:
                    gr.Markdown("#### Oscillator Parameters")
                    osc_n = gr.Slider(0, 10, 0, step=1, label="Quantum Number n")
                    osc_freq = gr.Slider(1, 100, 30, label="Frequency (THz)")
                
                with gr.Group(visible=False) as hydrogen_group:
                    gr.Markdown("#### Atomic Parameters")
                    h_n = gr.Slider(1, 4, 1, step=1, label="Principal Quantum Number n")
                    h_l = gr.Slider(0, 3, 0, step=1, label="Angular Momentum l")
                    h_m = gr.Slider(-3, 3, 0, step=1, label="Magnetic Quantum Number m")
                
                # Update visibility (unchanged)
                def update_controls(eq_type):
                    return (
                        gr.update(visible=eq_type=="Quantum Tunneling (Static)"),
                        gr.update(visible=eq_type=="Gaussian Wave Packet (Time-Dependent)"),
                        gr.update(visible=eq_type=="2D Tunneling with Angle"),
                        gr.update(visible=eq_type=="Particle in a Box"),
                        gr.update(visible=eq_type=="Harmonic Oscillator"),
                        gr.update(visible=eq_type=="Hydrogen Atom")
                    )
                
                equation_type.change(
                    update_controls,
                    [equation_type],
                    [tunneling_group, gaussian_group, angle_group, box_group, oscillator_group, hydrogen_group]
                )
                
                # TIME SLIDER
                time_slider = gr.Slider(0, 50, 0, step=1, label="Time Evolution Frame", 
                                      visible=False, interactive=True)
                
                # CALCULATE BUTTON (UNCHANGED)
                calculate_btn = gr.Button("🚀 Calculate Quantum State", elem_classes="calculate-btn")
                
                # RESULTS (UNCHANGED)
                gr.Markdown("### 📊 Results")
                result_display = gr.Markdown("### Configure quantum system to begin simulation")
                plot_output = gr.Plot()
                
                # INSIGHTS (ENHANCED)
                with gr.Accordion("💡 Physical Insight & Applications", open=True):
                    insight_display = gr.Markdown("""
                    <div class="application-highlight">
                        <strong style="color: #ffc107; font-size: 1.3em;">🔬 Real-World Applications</strong><br>
                        Select a quantum system to see how this physics enables cutting-edge technologies
                    </div>
                    """)
                
                gr.HTML('</div>')
        
        # CALCULATION HANDLER (FIXED - Proper backslash handling)
        def calculate_quantum_wrapper(
            particle_name, custom_mass_val, equation_type,
            energy_val, barrier_h_val, barrier_w_val,
            gaussian_sigma_val, gaussian_k0_val, gaussian_x0_val, gaussian_tmax_val,
            angle_val, energy_2d_val, barrier_h_2d_val, barrier_w_2d_val,
            box_n_val, box_width_val,
            osc_n_val, osc_freq_val,
            h_n_val, h_l_val, h_m_val,
            time_frame
        ):
            # Get particle parameters (unchanged)
            if particle_name == "Custom":
                particle_params = get_particle_params("Custom")
                particle_params["mass"] = custom_mass_val
            else:
                particle_params = get_particle_params(particle_name)
            
            m = particle_params["mass"]
            result_text = ""
            fig = go.Figure()
            insight = ""
            time_slider_update = gr.update(visible=False)
            
            try:
                engine = QuantumEngine()
                
                # Get equation metadata FIRST (new feature)
                equation_meta = get_equation_metadata(equation_type)
                
                # Build equation display panel (FIXED: No backslash issues - raw strings used in metadata)
                equation_html = f"""
                <div class="equation-panel">
                    <h3 style="color: #64b5f6; margin-top: 0; text-align: center; font-size: 2.0rem; margin-bottom: 22px;">
                        {equation_meta['title']}
                    </h3>
                    <div style="text-align: center; font-size: 1.75rem; margin: 25px 0; line-height: 1.65; color: #bbdefb;">
                        {equation_meta['equation']}
                    </div>
                    <div style="text-align: center; font-size: 1.55rem; margin: 20px 0; line-height: 1.6; color: #90caf9; background: rgba(100, 181, 246, 0.08); padding: 15px; border-radius: 12px;">
                        {equation_meta['solution']}
                    </div>
                    <div style="background: rgba(40, 60, 100, 0.35); border-radius: 16px; padding: 22px; margin: 25px 0; 
                                border-left: 4px solid #64b5f6; border: 1px solid rgba(100, 181, 246, 0.2);">
                        <strong style="color: #90caf9; font-size: 1.25em;">Physical Interpretation:</strong><br>
                        <span style="color: #e0e0e0; font-size: 1.15em; line-height: 1.5;">{equation_meta['interpretation']}</span>
                    </div>
                    <div class="application-highlight">
                        <strong style="color: #ffc107; font-size: 1.35em;">💡 Real-World Application:</strong><br>
                        <span style="color: #e0e0e0; font-size: 1.15em; line-height: 1.5;">{equation_meta['application']}</span>
                    </div>
                </div>
                """
                
                # Solve physics (unchanged logic)
                if equation_type == "Quantum Tunneling (Static)":
                    E = energy_val * EV_TO_J
                    V = barrier_h_val * EV_TO_J
                    L = barrier_w_val * 1e-9
                    
                    T, wave_data = engine.solve_tunneling(E, V, L, m)
                    if wave_data is not None:
                        fig = create_tunneling_plot_dark(wave_data, {"barrier_width": barrier_w_val})
                    
                    result_text = f"""
                    ### Quantum Tunneling Probability
                    
                    <div style="display: flex; gap: 25px; margin: 30px 0; align-items: center; justify-content: center;">
                        <div class="result-card">
                            <div style="font-size: 1.4em; opacity: 0.9; margin-bottom: 16px; letter-spacing: 0.5px; color: #90caf9;">Quantum Result</div>
                            <div style="font-size: 3.8em; font-weight: 800; letter-spacing: -1.5px; margin: 10px 0; color: #64b5f6;">{T * 100:.5f}%</div>
                            <div style="font-size: 1.3em; opacity: 0.85; margin-top: 8px; font-weight: 300; color: #bbdefb;">
                                {particle_params['symbol']} Tunneling Probability
                            </div>
                        </div>
                        <div class="result-card">
                            <div style="font-size: 1.4em; opacity: 0.9; margin-bottom: 16px; letter-spacing: 0.5px; color: #9e9e9e;">Classical Prediction</div>
                            <div style="font-size: 3.8em; font-weight: 800; letter-spacing: -1.5px; margin: 10px 0; color: #bdbdbd;">0.000%</div>
                            <div style="font-size: 1.3em; opacity: 0.85; margin-top: 8px; font-weight: 300; color: #e0e0e0;">
                                Energy Barrier
                            </div>
                        </div>
                    </div>
                    """
                    # FIXED: Proper backslash handling using raw string concatenation
                    insight = r"💡 **Quantum Tunneling**: The exponential sensitivity to barrier width ($T \propto e^{-2\kappa L}$) enables technologies like flash memory (data storage), scanning tunneling microscopes (atomic imaging), and nuclear fusion in stars (proton tunneling through Coulomb barrier)."
                
                elif equation_type == "Gaussian Wave Packet (Time-Dependent)":
                    frames = engine.solve_gaussian_tunneling(
                        m, energy_val, barrier_h_val, barrier_w_val,
                        x0_nm=gaussian_x0_val,
                        k0=gaussian_k0_val,
                        sigma_nm=gaussian_sigma_val,
                        t_max=gaussian_tmax_val * 1e-15,
                        dt=1e-17
                    )
                    
                    if frames:
                        fig = create_gaussian_evolution_plot(frames, int(time_frame))
                        _, _, _, _, _, T_final, R_final = frames[-1]
                        
                        result_text = f"""
                        ### Gaussian Wave Packet Evolution
                        
                        <div class="result-card">
                            <div style="font-size: 1.45em; opacity: 0.9; margin-bottom: 18px; letter-spacing: 0.5px; color: #90caf9;">Final Transmission</div>
                            <div style="font-size: 4.0em; font-weight: 800; letter-spacing: -1.5px; margin: 10px 0; color: #64b5f6;">{T_final * 100:.2f}%</div>
                            <div style="font-size: 1.35em; opacity: 0.85; margin-top: 10px; font-weight: 300; color: #bbdefb;">
                                Wave Packet Tunneling Probability
                            </div>
                        </div>
                        """
                        insight = r"💡 **Time-Dependent Quantum Dynamics**: Unlike static solutions, Gaussian wave packets show how quantum particles *evolve* in time. The wave packet spreads due to dispersion relation ($\Delta x \Delta p \geq \hbar/2$), and part tunnels while part reflects — creating interference patterns that reveal the wave nature of matter."
                        time_slider_update = gr.update(visible=True, maximum=len(frames)-1)
                
                elif equation_type == "2D Tunneling with Angle":
                    T, X, Y, prob_2d, V_2d, theta = engine.solve_2d_tunneling(
                        energy_2d_val, barrier_h_2d_val, barrier_w_2d_val, angle_val, m
                    )
                    fig = create_2d_tunneling_plot(X, Y, prob_2d, V_2d, angle_val)
                    
                    result_text = f"""
                    ### 2D Quantum Tunneling at {angle_val}° Incidence
                    
                    <div style="display: flex; gap: 25px; margin: 30px 0; align-items: center; justify-content: center;">
                        <div class="result-card">
                            <div style="font-size: 1.4em; opacity: 0.9; margin-bottom: 16px; letter-spacing: 0.5px; color: #90caf9;">Transmission Probability</div>
                            <div style="font-size: 3.8em; font-weight: 800; letter-spacing: -1.5px; margin: 10px 0; color: #64b5f6;">{T * 100:.3f}%</div>
                            <div style="font-size: 1.3em; opacity: 0.85; margin-top: 8px; font-weight: 300; color: #bbdefb;">
                                At {angle_val}° Incidence
                            </div>
                        </div>
                    </div>
                    """
                    insight = r"💡 **Angular Dependence**: At oblique angles, only the momentum component *normal* to the barrier affects tunneling probability ($E_{\perp} = E\cos^2\theta$). This reduces the effective barrier height, significantly enhancing transmission."
                
                elif equation_type == "Particle in a Box":
                    L = box_width_val * 1e-9
                    E_ev, wave_data = engine.solve_particle_in_box(int(box_n_val), L, m)
                    if wave_data is not None:
                        x, psi, prob, potential, E_n = wave_data
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x*1e9, y=psi, name='Wave Function', 
                                               line=dict(color='#64b5f6', width=3),
                                               fill='tozeroy', fillcolor='rgba(100, 181, 246, 0.15)'))
                        fig.update_layout(height=450, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                                        font=dict(color='#e0e0e0'), title=f"Particle in a Box (n={int(box_n_val)})",
                                        xaxis_title="Position (nm)", yaxis_title="Wave Amplitude")
                    
                    result_text = f"""
                    ### Particle in a Box - Energy Level n={int(box_n_val)}
                    
                    <div class="result-card">
                        <div style="font-size: 1.45em; opacity: 0.9; margin-bottom: 18px; letter-spacing: 0.5px; color: #90caf9;">Energy Level</div>
                        <div style="font-size: 4.0em; font-weight: 800; letter-spacing: -1.5px; margin: 10px 0; color: #64b5f6;">{E_ev:.6f} eV</div>
                        <div style="font-size: 1.35em; opacity: 0.85; margin-top: 10px; font-weight: 300; color: #bbdefb;">
                            {particle_name} in {box_width_val} nm box
                        </div>
                    </div>
                    """
                    insight = "💡 **Quantization**: Energy levels in confined systems are discrete. This explains semiconductor quantum dots (color-tunable displays), nanowire electronics, and the optical properties of conjugated polymers in organic LEDs."
                
                elif equation_type == "Harmonic Oscillator":
                    omega = osc_freq_val * 1e12 * 2 * np.pi
                    E_ev, wave_data = engine.solve_harmonic_oscillator(int(osc_n_val), omega, m)
                    if wave_data is not None:
                        x, psi, prob, potential, E_n = wave_data
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x*1e9, y=psi, name='Wave Function', 
                                               line=dict(color='#64b5f6', width=3),
                                               fill='tozeroy', fillcolor='rgba(100, 181, 246, 0.15)'))
                        fig.update_layout(height=450, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                                        font=dict(color='#e0e0e0'), title=f"Harmonic Oscillator (n={int(osc_n_val)})",
                                        xaxis_title="Position (nm)", yaxis_title="Wave Amplitude")
                    
                    result_text = f"""
                    ### Quantum Harmonic Oscillator - State n={int(osc_n_val)}
                    
                    <div class="result-card">
                        <div style="font-size: 1.45em; opacity: 0.9; margin-bottom: 18px; letter-spacing: 0.5px; color: #90caf9;">Energy Level</div>
                        <div style="font-size: 4.0em; font-weight: 800; letter-spacing: -1.5px; margin: 10px 0; color: #64b5f6;">{E_ev:.6f} eV</div>
                        <div style="font-size: 1.35em; opacity: 0.85; margin-top: 10px; font-weight: 300; color: #bbdefb;">
                            {particle_name} oscillator at {osc_freq_val} THz
                        </div>
                    </div>
                    """
                    insight = r"💡 **Zero-Point Energy**: Even at absolute zero (n=0), the oscillator has energy $\frac{1}{2}\hbar\omega$ — a purely quantum phenomenon verified in molecular spectroscopy."
                
                elif equation_type == "Hydrogen Atom":
                    E_ev, wave_data = engine.solve_hydrogen_atom(int(h_n_val), int(h_l_val), int(h_m_val))
                    if wave_data is not None:
                        r, R, prob, potential = wave_data
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=r*1e10, y=R, name='Radial Wavefunction', 
                                               line=dict(color='#64b5f6', width=3)))
                        fig.update_layout(height=450, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                                        font=dict(color='#e0e0e0'), title=f"Hydrogen Atom (n={int(h_n_val)}, l={int(h_l_val)})",
                                        xaxis_title="Radius (Å)", yaxis_title="Wave Amplitude")
                    
                    orbital_names = {(1,0): "1s", (2,0): "2s", (2,1): "2p", (3,0): "3s", (3,1): "3p", (3,2): "3d"}
                    orbital = orbital_names.get((int(h_n_val), int(h_l_val)), f"n={h_n_val},l={h_l_val}")
                    
                    result_text = f"""
                    ### Hydrogen Atom - Orbital {orbital}
                    
                    <div class="result-card">
                        <div style="font-size: 1.45em; opacity: 0.9; margin-bottom: 18px; letter-spacing: 0.5px; color: #90caf9;">Binding Energy</div>
                        <div style="font-size: 4.0em; font-weight: 800; letter-spacing: -1.5px; margin: 10px 0; color: #64b5f6;">{abs(E_ev):.4f} eV</div>
                        <div style="font-size: 1.35em; opacity: 0.85; margin-top: 10px; font-weight: 300; color: #bbdefb;">
                            Quantum numbers: n={int(h_n_val)}, l={int(h_l_val)}, m={int(h_m_val)}
                        </div>
                    </div>
                    """
                    insight = "💡 **Atomic Structure**: The Schrödinger equation's exact solution for hydrogen validated quantum mechanics and explains atomic spectra, chemical bonding, and the periodic table."
                
                return equation_html, result_text, fig, insight, time_slider_update
                
            except Exception as e:
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)[:50]}", xref="paper", yref="paper",
                                       x=0.5, y=0.5, showarrow=False, 
                                       font=dict(size=16, color="#f44336"))
                error_html = f"""
                <div class="equation-panel">
                    <h3 style="color: #f44336; text-align: center; font-size: 1.8rem;">Calculation Error</h3>
                    <p style="text-align: center; color: #ff8a80; font-size: 1.2rem; margin-top: 15px;">{str(e)[:100]}</p>
                </div>
                """
                return error_html, f"## Error\n\n{str(e)}", error_fig, "", gr.update(visible=False)
        
        # Connect calculate button (5 outputs)
        calculate_btn.click(
            calculate_quantum_wrapper,
            inputs=[selected_particle, custom_mass_input, equation_type,
                   energy, barrier_h, barrier_w,
                   gaussian_sigma, gaussian_k0, gaussian_x0, gaussian_tmax,
                   angle_slider, energy_2d, barrier_h_2d, barrier_w_2d,
                   box_n, box_width,
                   osc_n, osc_freq,
                   h_n, h_l, h_m,
                   time_slider],
            outputs=[equation_display, result_display, plot_output, insight_display, time_slider]
        )
        
        # PRESET HANDLERS
        def preset_tunneling():
            return calculate_quantum_wrapper(
                "Electron", M_E, "Quantum Tunneling (Static)",
                4.9, 5.0, 0.15,
                0.2, 5e9, -3.0, 2.0,
                30, 4.0, 5.0, 0.3,
                1, 1.0,
                0, 30.0,
                1, 0, 0,
                0
            )
        
        def preset_box():
            return calculate_quantum_wrapper(
                "Electron", M_E, "Particle in a Box",
                4.0, 5.0, 0.3,
                0.2, 5e9, -3.0, 2.0,
                30, 4.0, 5.0, 0.3,
                2, 2.0,
                0, 30.0,
                1, 0, 0,
                0
            )
        
        def preset_oscillator():
            return calculate_quantum_wrapper(
                "Electron", M_E, "Harmonic Oscillator",
                4.0, 5.0, 0.3,
                0.2, 5e9, -3.0, 2.0,
                30, 4.0, 5.0, 0.3,
                1, 1.0,
                1, 50.0,
                1, 0, 0,
                0
            )
        
        def preset_hydrogen():
            return calculate_quantum_wrapper(
                "Electron", M_E, "Hydrogen Atom",
                4.0, 5.0, 0.3,
                0.2, 5e9, -3.0, 2.0,
                30, 4.0, 5.0, 0.3,
                1, 1.0,
                0, 30.0,
                1, 0, 0,
                0
            )
        
        btn_tunneling.click(preset_tunneling, 
                          outputs=[equation_display, result_display, plot_output, insight_display, time_slider])
        btn_box.click(preset_box, 
                     outputs=[equation_display, result_display, plot_output, insight_display, time_slider])
        btn_oscillator.click(preset_oscillator, 
                           outputs=[equation_display, result_display, plot_output, insight_display, time_slider])
        btn_hydrogen.click(preset_hydrogen, 
                         outputs=[equation_display, result_display, plot_output, insight_display, time_slider])
        
        # FOOTER (ENHANCED)
        gr.HTML("""
        <div style="text-align: center; padding: 30px; color: #9e9e9e; border-top: 1px solid rgba(100, 181, 246, 0.15); 
                    margin-top: 25px; font-size: 0.98em; background: linear-gradient(135deg, #162447, #1a1a2e);">
            <div style="font-weight: 600; margin-bottom: 8px; color: #64b5f6;">Universal Quantum Physics Calculator</div>
            <div style="margin-bottom: 6px; color: #bbdefb;">
                Every simulation displays its governing Schrödinger equation • Exact analytical solutions • NIST CODATA constants
            </div>
            <div style="color: #78909c; font-style: italic; max-width: 750px; margin: 10px auto 0;">
                This tool bridges abstract quantum mathematics with physical intuition through equation-enhanced visualizations.
            </div>
        </div>
        """)
    
    return app

# LAUNCH (UNCHANGED)
if __name__ == "__main__":
    app = create_professional_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )