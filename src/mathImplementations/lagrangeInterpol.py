import numpy as np

class LagrangeInterpolator:
    def __init__(self, xi, yi):
        """
        Inicializa o interpolador de Lagrange com os pontos xi e yi.
        """
        self.xi = xi
        self.yi = yi

    def _interpolate_point(self, xp):
        """
        Avalia o polinômio interpolador de Lagrange no ponto xp.
        """
        n = len(self.xi)
        yp = 0
        for i in range(n):
            term = self.yi[i]
            for j in range(n):
                if j != i:
                    term *= (xp - self.xi[j]) / (self.xi[i] - self.xi[j])
            yp += term
        return yp

    def get_interpolated_values(self, x_values):
        """
        Retorna os valores interpolados para cada ponto em x_values.
        """
        return np.array([self._interpolate_point(x) for x in x_values])
    
    def get_error(self, x_values, y_true):
        """
        Calcula o erro RMSE da interpolação de Lagrange.
        
        Parâmetros:
            x_values (array-like): Pontos onde a interpolação é avaliada.
            y_true (array-like): Valores reais correspondentes.
        
        Retorna:
            float: O erro médio quadrático (RMSE).
        """
        y_pred = self.get_interpolated_values(x_values)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

def main():
    import matplotlib.pyplot as plt
    from src.utils.helpers import load_data

    df = load_data("data/ice_cream.csv")
    df_sorted = df.sort_values(by='Temperature')

    xi = df_sorted['Temperature'].values
    yi = df_sorted['Revenue'].values

    # Seleciona um subconjunto de pontos para evitar um polinômio de grau muito alto
    num_pontos = 6  # por exemplo, 6 pontos (polinômio de grau 5)
    x_interpol = np.linspace(0, 45, num_pontos)
    y_interpol = []
    for x_val in x_interpol:
        idx = (np.abs(xi - x_val)).argmin()
        y_interpol.append(yi[idx])
    y_interpol = np.array(y_interpol)

    # Cria o objeto interpolador de Lagrange com os pontos selecionados
    interpolator = LagrangeInterpolator(x_interpol, y_interpol)

    # Cria um conjunto de pontos para avaliação do polinômio interpolador
    x_novos = np.linspace(0, 45, 200)
    y_lagrange = interpolator.get_interpolated_values(x_novos)
    
    # Calcula o erro usando os pontos originais
    error = interpolator.get_error(xi, yi)
    print(f"Erro RMSE da Interpolação de Lagrange: {error}")

    # Gráfico do Método de Lagrange
    plt.figure(figsize=(10, 6))
    plt.plot(xi, yi, 'o', label='Dados Originais', alpha=0.5)
    plt.plot(x_interpol, y_interpol, 'ks', markersize=8, label='Pontos Selecionados')
    plt.plot(x_novos, y_lagrange, 'b--', label='Interpolação Lagrange')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Vendas/Receita')
    plt.title(f'Interpolação de Lagrange (RMSE: {error:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()