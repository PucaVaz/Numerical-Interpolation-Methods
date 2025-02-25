import numpy as np
from sympy.abc import x,y
import matplotlib.pyplot as plt


class MMQ():
    def __init__(self, xi, yi) -> None:
        """
        Inicializa o modelo de Mínimos Quadrados (MMQ) com os pontos de dados.

        Parâmetros:
            xi : array-like
                Vetor de dados para a variável independente (x).
            yi : array-like
                Vetor de dados para a variável dependente (y).

        Após a inicialização, os coeficientes a (angular) e b (linear) são calculados.
        """
        self.xi = xi
        self.yi = yi 
        self.a, self.b = self.get_a_and_b(xi, yi)

    def _calculate_a(self, xi, yi):
        """
        Calcula o coeficiente angular (a) da reta de regressão pelos mínimos quadrados.
        
        A fórmula é:
            a = (n * Σ(xi * yi) - Σ(xi) * Σ(yi)) / (n * Σ(xi²) - (Σ(xi))²)
        
        Parâmetros:
            xi : array-like
                Vetor de dados para a variável independente.
            yi : array-like
                Vetor de dados para a variável dependente.
        
        Retorna:
            a : float
                Coeficiente angular da reta de regressão.
        """
        n = len(xi)
        return (n * np.sum(xi * yi) - np.sum(xi) * np.sum(yi)) / (n * np.sum(xi**2) - np.sum(xi)**2)

    def _calculate_b(self, xi, yi, a1):
        """
        Calcula o coeficiente linear (b) da reta de regressão usando a média dos valores.

        A fórmula é:
            b = média(yi) - a * média(xi)
        
        Parâmetros:
            xi : array-like
                Vetor de dados para a variável independente.
            yi : array-like
                Vetor de dados para a variável dependente.
            a1 : float
                Coeficiente angular previamente calculado.
        
        Retorna:
            b : float
                Coeficiente linear da reta de regressão.
        """
        n = len(xi)
        return (np.sum(yi) / n) - a1 * (np.sum(xi) / n)

    def get_a_and_b(self, xi, yi):
        """
        Calcula e retorna os coeficientes a e b da reta de regressão pelos mínimos quadrados.

        Utiliza os métodos internos _calculate_a e _calculate_b.

        Parâmetros:
            xi : array-like
                Vetor de dados para a variável independente.
            yi : array-like
                Vetor de dados para a variável dependente.
        
        Retorna:
            (a, b) : tuple
                Coeficiente angular (a) e coeficiente linear (b) da reta de regressão.
        """
        a = self._calculate_a(xi, yi)
        b = self._calculate_b(xi, yi, a)
        return a, b
    
    def get_mmq_linear(self):
        """
        Retorna a expressão simbólica para o polinômio linear de mínimos quadrados.

        A equação obtida é:
            y = b + a * x

        Retorna:
            y_expr : sympy expression
                Expressão simbólica representando a reta de regressão.
        """
        y_expr = self.b + self.a * x 
        return y_expr
    
    def get_mmq_quadratic(self):
        """
        Retorna a expressão simbólica para o polinômio quadrático de mínimos quadrados.

        Aqui, o coeficiente a está associado ao termo quadrático:
            y = b + a * x²

        Retorna:
            y_expr : sympy expression
                Expressão simbólica representando o polinômio quadrático.
        """
        y_expr = self.b + self.a * x ** 2
        return y_expr 

def main():
    """
    Função principal que:
      - Carrega os dados de "ice_cream.csv" utilizando o helper load_data.
      - Ordena os dados pela coluna "Temperature".
      - Extrai os arrays xi e yi correspondentes à temperatura e à receita/vendas.
      - Calcula os coeficientes de regressão utilizando o método dos mínimos quadrados (MMQ).
      - Imprime as expressões simbólicas para os modelos linear e quadrático.
    """
    from src.utils.helpers import load_data
    df = load_data("data/ice_cream.csv")

    # Ordena o DataFrame pela coluna "Temperature"
    df_sorted = df.sort_values(by='Temperature')

    # Extrai xi e yi do DataFrame ordenado
    xi = df_sorted['Temperature'].values
    yi = df_sorted['Revenue'].values

    # Cria o objeto MMQ para calcular os coeficientes de regressão
    mmq = MMQ(xi, yi)

    # Obtém os modelos de MMQ linear e quadrático (expressões simbólicas)
    y_linear = mmq.get_mmq_linear()
    y_quad = mmq.get_mmq_quadratic()
    
    print(f"Linear MMQ Result: {y_linear}, Quadratic MMQ Result: {y_quad}")

    y_linear = mmq.get_mmq_linear()
    # Plot the Linear Graphic
    plt.figure(figsize=(8, 6))
    plt.scatter(xi, yi, color='blue', label='Data Points')
    plt.plot(xi, y_linear, color='red', label='Linear MMQ')
    plt.xlabel('Temperature')
    plt.ylabel('Revenue')
    plt.title('Linear (MMQ)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()