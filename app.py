import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
st.set_page_config(page_title="Complex Desmos", layout="wide")
st.markdown("""
 <style>
     .css-1d391kg, .css-1v0mbdj {background-color: #0e1117; color: white;}
     .stPlotlyChart {background-color: #0e1117;}
     .stTextInput > div > div > input {color: white; background-color: #262730;}
 </style>
""",
            unsafe_allow_html=True)
st.title("Complex Desmos")
with st.sidebar:
    st.header("Function")
    expr_str = st.text_input("f(x):", value="")
    st.subheader("View")
    view_mode = st.radio(
        "Mode:",
        ["2D: x vs y", "2D: I vs y", "3D: x-I-y", "3D+I (Re+Im)"],
        index=0
    )
    st.markdown("---")
    st.subheader("Cubic Zoom")
    zoom = st.slider("Zoom Level (Cube Size)", 5.0, 40.0, 5.0, 5.0)
    domain = [-zoom, zoom]
    x_min, x_max = domain
    i_min, i_max = domain
    y_min, y_max = domain
    st.markdown("---")
    st.subheader("Plane Opacity")
    plane_opacity = st.slider("Opacity for 3D planes", 0.0, 1.0, 1.0, 0.05)
    st.markdown("---")
    st.subheader("3D Planes")
    show_real_plane = st.checkbox("Red: I = 0 (real axis)", value=False)
    show_imag_plane = st.checkbox("Green: x = 0 (imag axis)", value=False)
    show_zero_plane = st.checkbox("Dark: y = 0 (Re(f)=0)", value=False)
    if view_mode == "3D+I (Re+Im)":
        st.markdown("---")
        st.subheader("Surface Toggle")
        show_re_surface = st.checkbox("Show Re(f) surface", value=True)
        show_im_surface = st.checkbox("Show Im(f) surface", value=True)
    else:
        show_re_surface = True
        show_im_surface = True
res = 140
x_sym = sp.Symbol('x')
if expr_str.strip() == "":
    st.warning("Enter a function above to begin plotting.")
    st.stop()
try:
    expr = parse_expr(expr_str,
                      transformations=standard_transformations +
                      (implicit_multiplication, ))
    st.success(f"f(x) = {expr}")
    f_np = sp.lambdify(x_sym, expr, 'numpy')
except Exception as e:
    st.error(f"Parse error: {e}")
    st.stop()


def find_exact_zeros_sympy(expr, x_min, x_max, i_min, i_max):
    z = sp.Symbol('z')
    expr_z = expr.subs(x_sym, z)
    equation = sp.Eq(expr_z, 0)
    zeros = []
    try:
        solutions = sp.solve(equation, z)
        for sol in solutions:
            if sol.is_complex:
                x_val = float(sp.re(sol))
                i_val = float(sp.im(sol))
                if x_min <= x_val <= x_max and i_min <= i_val <= i_max:
                    y_val = float(sp.re(expr_z.subs(z, sol)))
                    zeros.append((x_val, y_val, i_val))
    except Exception:
        pass
    return zeros


zeros = find_exact_zeros_sympy(expr, x_min, x_max, i_min, i_max)


def display_zeros_table(zeros):
    if not zeros:
        st.info("**No zeros found in the current domain.**")
        return

    st.markdown("---")
    st.subheader("**Exact Zeros Found**")

    zero_data = []
    for i, (x_val, y_val, i_val) in enumerate(zeros, 1):
        complex_z = complex(x_val, i_val)
        zero_data.append({
            "Zero #": i,
            "z = x + i·I": f"{x_val:.6f} + {i_val:.6f}i",
            "Re(f(z))": f"{y_val:.6f}",
            "Im(f(z))": "0.000000"
        })

    st.table(zero_data)

    st.markdown("*Zeros where f(z) = 0 exactly*")


if view_mode == "2D: x vs y":
    xs = np.linspace(x_min, x_max, 1000)
    ys = np.real(f_np(xs))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=xs,
                   y=ys,
                   mode='lines',
                   name='Re(f(x))',
                   line=dict(color='cyan')))
    fig.add_hline(y=0, line_color='red', line_dash='dash', line_width=2)
    if zeros:
        zx, zy, zi = zip(*zeros)
        zx_real = [x for x, i in zip(zx, zi) if abs(i) < 1e-9]
        zy_real = [y for x, y, i in zip(zx, zy, zi) if abs(i) < 1e-9]
        if zx_real:
            fig.add_trace(
                go.Scatter(x=zx_real,
                           y=zy_real,
                           mode='markers',
                           name='Zeros',
                           marker=dict(color='black',
                                       size=10,
                                       symbol='circle',
                                       line=dict(color='white', width=1))))
    fig.update_layout(template="plotly_dark",
                      title="2D: x vs y",
                      xaxis_title="x (Real Input)",
                      yaxis_title="y = Re(f)",
                      height=600)
    st.plotly_chart(fig, use_container_width=True)

    display_zeros_table(zeros)

elif view_mode == "2D: I vs y":
    is_ = np.linspace(i_min, i_max, 1000)
    ys = np.real(f_np(1j * is_))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=is_,
                   y=ys,
                   mode='lines',
                   name='Re(f(iI))',
                   line=dict(color='magenta')))
    fig.add_hline(y=0, line_color='red', line_dash='dash', line_width=2)
    if zeros:
        zx, zy, zi = zip(*zeros)
        zi_imag = [i for x, i in zip(zx, zi) if abs(x) < 1e-9]
        zy_imag = [y for x, y, i in zip(zx, zy, zi) if abs(x) < 1e-9]
        if zi_imag:
            fig.add_trace(
                go.Scatter(x=zi_imag,
                           y=zy_imag,
                           mode='markers',
                           name='Zeros',
                           marker=dict(color='black',
                                       size=10,
                                       symbol='circle',
                                       line=dict(color='white', width=1))))
    fig.update_layout(template="plotly_dark",
                      title="2D: I vs y",
                      xaxis_title="I (Imag Input)",
                      yaxis_title="y = Re(f)",
                      height=600)
    st.plotly_chart(fig, use_container_width=True)

    display_zeros_table(zeros)

elif view_mode == "3D: x-I-y":
    X = np.linspace(x_min, x_max, res)
    I = np.linspace(i_min, i_max, res)
    Xg, Ig = np.meshgrid(X, I)
    Zg = Xg + 1j * Ig
    Yg = np.real(f_np(Zg))
    margin = int(0.03 * res)
    trim = slice(margin, res - margin)
    Xg_t = Xg[trim, trim]
    Ig_t = Ig[trim, trim]
    Yg_t = Yg[trim, trim]
    contour_step = (y_max - y_min) / 20
    fig = go.Figure()
    fig.add_trace(
        go.Surface(x=Xg_t,
                   y=Yg_t,
                   z=Ig_t,
                   colorscale='RdBu',
                   showscale=True,
                   colorbar=dict(title="Re(f)", x=0.88),
                   lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                   lightposition=dict(x=0, y=0, z=10000),
                   opacity=1.0,
                   contours=dict(x=dict(show=True, color='white', width=1),
                                 y=dict(show=True, color='white', width=1),
                                 z=dict(show=True,
                                        start=y_min,
                                        end=y_max,
                                        size=contour_step,
                                        color='white',
                                        width=1.8))))
    if show_real_plane:
        Yp = np.linspace(y_min, y_max, res)
        Xp, Yp = np.meshgrid(X, Yp)
        fig.add_trace(
            go.Surface(x=Xp,
                       y=Yp,
                       z=np.zeros_like(Xp),
                       colorscale=[[0, '#8B0000'], [1, '#8B0000']],
                       showscale=False,
                       opacity=plane_opacity,
                       surfacecolor=np.zeros_like(Xp),
                       contours=dict(x=dict(show=True, color='white', width=1),
                                     y=dict(show=True, color='white',
                                            width=1))))
    if show_imag_plane:
        Yp = np.linspace(y_min, y_max, res)
        Ip, Yp = np.meshgrid(I, Yp)
        fig.add_trace(
            go.Surface(x=np.zeros_like(Ip),
                       y=Yp,
                       z=Ip,
                       colorscale=[[0, '#006400'], [1, '#006400']],
                       showscale=False,
                       opacity=plane_opacity,
                       surfacecolor=np.zeros_like(Ip),
                       contours=dict(y=dict(show=True, color='white', width=1),
                                     z=dict(show=True, color='white',
                                            width=1))))
    if show_zero_plane:
        fig.add_trace(
            go.Surface(x=Xg_t,
                       y=np.zeros_like(Xg_t),
                       z=Ig_t,
                       colorscale=[[0, '#1a1a1a'], [1, '#1a1a1a']],
                       showscale=False,
                       opacity=plane_opacity,
                       surfacecolor=np.zeros_like(Xg_t),
                       contours=dict(x=dict(show=True, color='white', width=1),
                                     z=dict(show=True, color='white',
                                            width=1))))
    if zeros:
        zx, zy, zi = zip(*zeros)
        fig.add_trace(
            go.Scatter3d(
                x=zx,
                y=zy,
                z=zi,
                mode='markers',
                name='Exact Zeros',
                marker=dict(color='black',
                            size=6,
                            symbol='circle',
                            line=dict(color='white', width=1)),
                hovertemplate=
                '<b>Zero</b><br>x=%{x:.6f}<br>y=%{y:.6f}<br>I=%{z:.6f}<extra></extra>'
            ))
    fig.update_layout(template="plotly_dark",
                      title=f"3D View: f(x + iI) = {expr}",
                      scene=dict(xaxis_title="x (Real Input)",
                                 yaxis_title="y = Re(f) (UP)",
                                 zaxis_title="I (Imag Input)",
                                 camera=dict(eye=dict(x=1.6, y=1.8, z=1.2)),
                                 aspectmode='cube',
                                 xaxis=dict(range=domain,
                                            showgrid=True,
                                            zeroline=True,
                                            zerolinecolor='white'),
                                 yaxis=dict(range=domain,
                                            showgrid=True,
                                            zeroline=True,
                                            zerolinecolor='red'),
                                 zaxis=dict(range=domain,
                                            showgrid=True,
                                            zeroline=True,
                                            zerolinecolor='white'),
                                 bgcolor='#0e1117'),
                      height=800,
                      legend=dict(x=0.7,
                                  y=0.95,
                                  bgcolor='rgba(0,0,0,0.6)',
                                  font=dict(color='white')))
    st.plotly_chart(fig, use_container_width=True)

    display_zeros_table(zeros)

else: 
    X = np.linspace(x_min, x_max, res)
    I = np.linspace(i_min, i_max, res)
    Xg, Ig = np.meshgrid(X, I)
    Zg = Xg + 1j * Ig
    Y_real = np.real(f_np(Zg))
    Y_imag = np.imag(f_np(Zg))
    margin = int(0.03 * res)
    trim = slice(margin, res - margin)
    Xg_t = Xg[trim, trim]
    Ig_t = Ig[trim, trim]
    Y_real_t = Y_real[trim, trim]
    Y_imag_t = Y_imag[trim, trim]
    contour_step = (y_max - y_min) / 20
    fig = go.Figure()
    if show_re_surface:
        fig.add_trace(
            go.Surface(x=Xg_t,
                       y=Y_real_t,
                       z=Ig_t,
                       colorscale='RdBu',
                       showscale=True,
                       colorbar=dict(title="Re(f)", x=0.88),
                       lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                       lightposition=dict(x=0, y=0, z=10000),
                       opacity=1.0,
                       contours=dict(x=dict(show=True, color='white', width=1),
                                     y=dict(show=True, color='white', width=1),
                                     z=dict(show=True,
                                            start=y_min,
                                            end=y_max,
                                            size=contour_step,
                                            color='white',
                                            width=1.8)),
                       name="Re(f)"))
    if show_im_surface:
        fig.add_trace(
            go.Surface(x=Xg_t,
                       y=Y_imag_t,
                       z=Ig_t,
                       colorscale='YlOrRd',
                       showscale=True,
                       colorbar=dict(title="Im(f)", x=1.0),
                       lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                       lightposition=dict(x=0, y=0, z=10000),
                       opacity=1.0,
                       contours=dict(x=dict(show=True, color='white', width=1),
                                     y=dict(show=True, color='white', width=1),
                                     z=dict(start=y_min,
                                            end=y_max,
                                            size=contour_step,
                                            color='white',
                                            width=1.8)),
                       name="Im(f)"))
    # Approximate zero set (only if both surfaces are on)
    if show_re_surface and show_im_surface:
        tol = 0.15 * (y_max - y_min) / res
        zero_mask = (np.abs(Y_real_t) < tol) & (np.abs(Y_imag_t) < tol)
        if np.any(zero_mask):
            fig.add_trace(
                go.Scatter3d(
                    x=Xg_t[zero_mask],
                    y=Y_real_t[zero_mask],
                    z=Ig_t[zero_mask],
                    mode='markers',
                    name='Approx Zeros',
                    marker=dict(color='yellow', size=3),
                    hovertemplate=
                    '<b>Approx Zero</b><br>x=%{x:.3f}<br>Re=%{y:.3f}<br>Im~0<br>I=%{z:.3f}<extra></extra>'
                ))
    if zeros:
        zx, zy, zi = zip(*zeros)
        fig.add_trace(
            go.Scatter3d(
                x=zx,
                y=zy,
                z=zi,
                mode='markers',
                name='Exact Zeros',
                marker=dict(color='black',
                            size=6,
                            symbol='circle',
                            line=dict(color='white', width=1)),
                hovertemplate=
                '<b>Exact Zero</b><br>x=%{x:.6f}<br>y=%{y:.6f}<br>I=%{z:.6f}<extra></extra>'
            ))
    if show_real_plane:
        Yp = np.linspace(y_min, y_max, res)
        Xp, Yp = np.meshgrid(X, Yp)
        fig.add_trace(
            go.Surface(x=Xp,
                       y=Yp,
                       z=np.zeros_like(Xp),
                       colorscale=[[0, '#8B0000'], [1, '#8B0000']],
                       showscale=False,
                       opacity=plane_opacity,
                       surfacecolor=np.zeros_like(Xp),
                       contours=dict(x=dict(show=True, color='white', width=1),
                                     y=dict(show=True, color='white',
                                            width=1))))
    if show_imag_plane:
        Yp = np.linspace(y_min, y_max, res)
        Ip, Yp = np.meshgrid(I, Yp)
        fig.add_trace(
            go.Surface(x=np.zeros_like(Ip),
                       y=Yp,
                       z=Ip,
                       colorscale=[[0, '#006400'], [1, '#006400']],
                       showscale=False,
                       opacity=plane_opacity,
                       surfacecolor=np.zeros_like(Ip),
                       contours=dict(y=dict(show=True, color='white', width=1),
                                     z=dict(show=True, color='white',
                                            width=1))))
    if show_zero_plane:
        fig.add_trace(
            go.Surface(x=Xg_t,
                       y=np.zeros_like(Xg_t),
                       z=Ig_t,
                       colorscale=[[0, '#1a1a1a'], [1, '#1a1a1a']],
                       showscale=False,
                       opacity=plane_opacity,
                       surfacecolor=np.zeros_like(Xg_t),
                       contours=dict(x=dict(show=True, color='white', width=1),
                                     z=dict(show=True, color='white',
                                            width=1))))
    fig.update_layout(
        template="plotly_dark",
        title=f"3D+I View – Re(f) (RdBu) + Im(f) (YlOrRd) – {expr}",
        scene=dict(xaxis_title="x (Real Input)",
                   yaxis_title="y = f (UP)",
                   zaxis_title="I (Imag Input)",
                   camera=dict(eye=dict(x=1.6, y=1.8, z=1.2)),
                   aspectmode='cube',
                   xaxis=dict(range=domain,
                              showgrid=True,
                              zeroline=True,
                              zerolinecolor='white'),
                   yaxis=dict(range=domain,
                              showgrid=True,
                              zeroline=True,
                              zerolinecolor='red'),
                   zaxis=dict(range=domain,
                              showgrid=True,
                              zeroline=True,
                              zerolinecolor='white'),
                   bgcolor='#0e1117'),
        height=800,
        legend=dict(x=0.01,
                    y=0.99,
                    bgcolor='rgba(0,0,0,0.6)',
                    font=dict(color='white')))
    st.plotly_chart(fig, use_container_width=True)

    display_zeros_table(zeros)
