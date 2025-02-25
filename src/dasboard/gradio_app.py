import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import sympy
from sympy.abc import x

from src.mathImplementations.mmq import MMQ
from src.mathImplementations.newtonInterpol import NewtonInterpolator
from src.mathImplementations.lagrangeInterpol import LagrangeInterpolator
from src.utils.helpers import load_data 

def fig_to_img(fig):
    """
    Converts a Matplotlib figure to a PIL Image.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img

def load_and_sort_data():
    """
    Loads the ice cream data using the helper and sorts it by Temperature.
    
    Returns:
        xi (np.array): Temperature values.
        yi (np.array): Revenue values.
    """
    df = load_data("data/ice_cream.csv")
    df_sorted = df.sort_values(by='Temperature')
    xi = df_sorted['Temperature'].values
    yi = df_sorted['Revenue'].values
    return xi, yi

def generate_mmq_plot(model_type="Linear"):
    """
    Generates a plot for the MMQ (Least Squares) model.

    Parameters:
        model_type (str): "Linear" or "Quadratic".
    
    Returns:
        PIL Image: The regression plot.
    """
    xi, yi = load_and_sort_data()
    
    # Create MMQ instance and select the model type.
    mmq = MMQ(xi, yi)
    if model_type == "Linear":
        y_expr = mmq.get_mmq_linear()
        title = "Regressão Linear por Mínimos Quadrados"
    else:  # Quadratic
        y_expr = mmq.get_mmq_quadratic()
        title = "Regressão Quadrática por Mínimos Quadrados"
    
    # Convert the symbolic expression to a numerical function.
    f_model = sympy.lambdify(x, y_expr, "numpy")
    y_model = f_model(xi)
    
    # Plot the original data and the model.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xi, yi, color='blue', label='Pontos de Dados')
    ax.plot(xi, y_model, color='red', label=("Linear" if model_type == "Linear" else "Quadrático") + " MMQ")
    ax.set_xlabel("Temperatura")
    ax.set_ylabel("Receita")
    ax.set_title(title)
    ax.legend()
    
    img = fig_to_img(fig)
    plt.close(fig)
    return img

def generate_newton_plot(num_nodes=6):
    """
    Generates a plot using Newton's Interpolation.

    Parameters:
        num_nodes (int): Number of interpolation nodes.
    
    Returns:
        PIL Image: The Newton interpolation plot.
    """
    xi, yi = load_and_sort_data()
    
    x_interp = np.linspace(np.min(xi), np.max(xi), num_nodes)
    y_interp = []
    for x_val in x_interp:
        idx = (np.abs(xi - x_val)).argmin()
        y_interp.append(yi[idx])
    y_interp = np.array(y_interp)
    
    # Cria interpolador de Newton e avalia em uma grade fina
    interpolator = NewtonInterpolator(x_interp, y_interp)
    x_new = np.linspace(np.min(xi), np.max(xi), 200)
    y_newton = interpolator.get_interpolated_values(x_new)
    
    # Calcula o erro RMSE usando todos os pontos de dados
    error = interpolator.get_error(xi, yi)
    
    # Plota os dados e a interpolação
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xi, yi, 'o', label='Dados Originais', alpha=0.5)
    ax.plot(x_interp, y_interp, 'ks', markersize=8, label='Nós Selecionados')
    ax.plot(x_new, y_newton, 'r-', label='Interpolação de Newton')
    ax.set_xlabel("Temperatura")
    ax.set_ylabel("Receita")
    ax.set_title(f"Interpolação de Newton (RMSE: {error:.2f})")
    ax.legend()
    ax.grid(True)
    
    img = fig_to_img(fig)
    plt.close(fig)
    return img

def generate_lagrange_plot(num_nodes=6):
    xi, yi = load_and_sort_data()
    
    # Seleciona nós de interpolação
    x_interp = np.linspace(np.min(xi), np.max(xi), num_nodes)
    y_interp = []
    for x_val in x_interp:
        idx = (np.abs(xi - x_val)).argmin()
        y_interp.append(yi[idx])
    y_interp = np.array(y_interp)
    
    # Cria interpolador de Lagrange e avalia em uma grade fina
    interpolator = LagrangeInterpolator(x_interp, y_interp)
    x_new = np.linspace(np.min(xi), np.max(xi), 200)
    y_lagrange = interpolator.get_interpolated_values(x_new)
    
    # Calcula o erro RMSE usando os pontos do conjunto de dados original
    error = interpolator.get_error(xi, yi)
    
    # Plota os dados e a interpolação
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xi, yi, 'o', label='Dados Originais', alpha=0.5)
    ax.plot(x_interp, y_interp, 'ks', markersize=8, label='Nós Selecionados')
    ax.plot(x_new, y_lagrange, 'b--', label='Interpolação de Lagrange')
    ax.set_xlabel("Temperatura")
    ax.set_ylabel("Receita")
    ax.set_title(f"Interpolação de Lagrange (RMSE: {error:.2f})")
    ax.legend()
    ax.grid(True)
    
    img = fig_to_img(fig)
    plt.close(fig)
    return img

def generate_dashboard_plot(method, num_nodes):
    """
    Main function for the Gradio dashboard.
    
    Parameters:
        method (str): Selected method. Options: "MMQ Linear", "MMQ Quadratic", 
                      "Newton Interpolation", "Lagrange Interpolation".
        num_nodes (int): Number of interpolation nodes (only used for Newton/Lagrange).
    
    Returns:
        PIL Image: The generated plot image.
    """
    if method == "MMQ Linear":
        return generate_mmq_plot("Linear")
    elif method == "MMQ Quadrático":
        return generate_mmq_plot("Quadratic")
    elif method == "Interpolação de Newton":
        return generate_newton_plot(num_nodes)
    elif method == "Interpolação de Lagrange":
        return generate_lagrange_plot(num_nodes)
    else:
        raise ValueError("Método desconhecido selecionado.")

def main():
    # Define inputs for the Gradio interface.
    method_input = gr.Dropdown(
        choices=["MMQ Linear", "MMQ Quadrático", "Interpolação de Newton", "Interpolação de Lagrange"],
        label="Selecione o Método Matemático",
    )
    num_nodes_input = gr.Slider(
        minimum=3, maximum=20, step=1,
        label="Número de Nós de Interpolação (para Newton/Lagrange)"
    )
    
    # Create the interface.
    interface = gr.Interface(
        fn=generate_dashboard_plot,
        inputs=[method_input, num_nodes_input],
        outputs="image",
        title="Dashboard de Implementações Matemáticas",
        description="Explore vários modelos matemáticos (regressão por Mínimos Quadrados e Interpolação) usando o conjunto de dados de vendas de sorvete."
    )
    interface.launch()

if __name__ == "__main__":
    main()
