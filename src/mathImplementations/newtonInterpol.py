import numpy as np

class NewtonInterpolator:
    def __init__(self, x_nodes, y_nodes) -> None:
        """
        Inicializa o interpolador de Newton com os nós de interpolação e valores correspondentes.
        Calcula os coeficientes de diferenças divididas.
        """
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.coef = self.newton_divided_differences(x_nodes, y_nodes)

    def newton_divided_differences(self, x, y):
        """
        Calcula os coeficientes do polinômio de Newton usando diferenças divididas.
        
        Parâmetros:
            x : array com os nós de interpolação.
            y : array com os valores correspondentes.
        
        Retorna:
            coef : array com os coeficientes (termos das diferenças divididas).
        """
        n = len(x)
        coef = np.copy(y).astype(float)
        for j in range(1, n):
            coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
        return coef

    def newton_interpolate(self, x_val):
        """
        Avalia o polinômio de Newton em um ponto (ou vetor) x_val.
        
        Parâmetros:
            x_val : ponto (ou array de pontos) onde avaliar o polinômio.
        
        Retorna:
            Valor interpolado.
        """
        if np.isscalar(x_val):
            result = self.coef[-1]
            for j in range(len(self.coef) - 2, -1, -1):
                result = result * (x_val - self.x_nodes[j]) + self.coef[j]
            return result
        else:
            x_val = np.asarray(x_val)
            result = np.zeros_like(x_val, dtype=float)
            for i, xp in enumerate(x_val):
                temp = self.coef[-1]
                for j in range(len(self.coef) - 2, -1, -1):
                    temp = temp * (xp - self.x_nodes[j]) + self.coef[j]
                result[i] = temp
            return result

    def get_interpolated_values(self, x_values):
        """
        Retorna os valores interpolados para os pontos em x_values.
        """
        return self.newton_interpolate(x_values)
    
    def get_error(self, x_values, y_true):
        """
        Calcula o erro RMSE da interpolação de Newton.
        
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

    # Seleciona um subconjunto de pontos para evitar polinômio de grau muito alto
    num_pontos = 6  # Por exemplo, 6 pontos (polinômio de grau 5)
    x_interpol = np.linspace(0, 45, num_pontos)
    y_interpol = []
    for x_val in x_interpol:
        idx = (np.abs(xi - x_val)).argmin()
        y_interpol.append(yi[idx])
    y_interpol = np.array(y_interpol)

    # Cria o objeto interpolador de Newton com os pontos selecionados
    interpolator = NewtonInterpolator(x_interpol, y_interpol)

    # Cria um conjunto de pontos para avaliação do polinômio interpolador
    x_novos = np.linspace(0, 45, 200)
    y_newton = interpolator.get_interpolated_values(x_novos)
    
    # Calcula o erro usando os pontos originais
    error = interpolator.get_error(xi, yi)
    print(f"Erro RMSE da Interpolação de Newton: {error}")

    # Gráfico do Método de Newton
    plt.figure(figsize=(10, 6))
    plt.plot(xi, yi, 'o', label='Dados Originais', alpha=0.5)
    plt.plot(x_interpol, y_interpol, 'ks', markersize=8, label='Pontos Selecionados')
    plt.plot(x_novos, y_newton, 'r-', label='Interpolação Newton')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Vendas/Receita')
    plt.title(f'Interpolação de Newton (RMSE: {error:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()