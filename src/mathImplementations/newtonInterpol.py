import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suponha que seu DataFrame já está carregado e tem as colunas 'Temperature' e 'Revenue'
# Exemplo:
# df = pd.read_csv('seu_arquivo.csv')

# Ordene os dados se necessário
# Ordenar todo o DataFrame pela coluna 'Temperature'
df_sorted = df.sort_values(by='Temperature')

# Agora, extrair xi e yi do df_sorted
xi = df_sorted['Temperature'].values
yi = df_sorted['Revenue'].values

# =============================================================================
# 1. Selecionando um subconjunto de pontos para evitar polinômio de grau alto
# =============================================================================
# Para interpolar no intervalo [0, 45] com poucos pontos, podemos escolher pontos uniformemente espaçados.
num_pontos = 6  # por exemplo, 6 pontos (grau 5 no polinômio)
x_interpol = np.linspace(0, 45, num_pontos)

# Para cada valor em x_interpol, pegamos o valor de y mais próximo no nosso dataset
y_interpol = []
for x_val in x_interpol:
    idx = (np.abs(xi - x_val)).argmin()
    y_interpol.append(yi[idx])
y_interpol = np.array(y_interpol)



# =============================================================================
# 2. Método de Newton (Diferenças Divididas)
# =============================================================================

def newton_divided_differences(x, y):
    """
    Calcula os coeficientes do polinômio de Newton usando diferenças divididas.
    
    Parâmetros:
      x : array com os nós de interpolação.
      y : array com os valores correspondentes.
      
    Retorna:
      coef : array com os coeficientes (os termos das diferenças divididas).
    """
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

def newton_interpolate(x_nodes, coef, x_val):
    """
    Avalia o polinômio de Newton em um ponto (ou vetor) x_val.
    
    Parâmetros:
      x_nodes : array com os nós de interpolação.
      coef    : array com os coeficientes obtidos por newton_divided_differences.
      x_val   : ponto (ou array de pontos) onde avaliar o polinômio.
    
    Retorna:
      valor interpolado em x_val.
    """
    # Se x_val for um array, inicializamos um array de resultados
    if np.isscalar(x_val):
        result = coef[-1]
        for j in range(len(coef) - 2, -1, -1):
            result = result * (x_val - x_nodes[j]) + coef[j]
        return result
    else:
        x_val = np.asarray(x_val)
        result = np.zeros_like(x_val, dtype=float)
        for i, xp in enumerate(x_val):
            temp = coef[-1]
            for j in range(len(coef) - 2, -1, -1):
                temp = temp * (xp - x_nodes[j]) + coef[j]
            result[i] = temp
        return result

# Calcula os coeficientes para o conjunto reduzido de pontos
coef = newton_divided_differences(x_interpol, y_interpol)

# Cria um conjunto de pontos para avaliação dos polinômios interpoladores
x_novos = np.linspace(0, 45, 200)

y_newton = newton_interpolate(x_interpol, coef, x_novos)

# --- Gráfico do Método de Newton ---
plt.figure(figsize=(10, 6))
plt.plot(xi, yi, 'o', label='Dados Originais', alpha=0.5)
plt.plot(x_interpol, y_interpol, 'ks', markersize=8, label='Pontos Selecionados')
plt.plot(x_novos, y_newton, 'r-', label='Interpolação Newton')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas/Receita')
plt.title('Interpolação de Newton')
plt.legend()
plt.grid(True)
plt.show()