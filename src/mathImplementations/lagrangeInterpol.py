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



# ---------------------------------------------------------------------------
# 3. Método de Lagrange
# ---------------------------------------------------------------------------
def lagrange_interpolation(x, y, xp):
    """
    Avalia o polinômio interpolador de Lagrange no ponto xp.
    """
    n = len(x)
    yp = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (xp - x[j]) / (x[i] - x[j])
        yp += term
    return yp


# Cria um conjunto de pontos para avaliação dos polinômios interpoladores
x_novos = np.linspace(0, 45, 200)
# Avalia a interpolação usando o método de Lagrange
y_lagrange = np.array([lagrange_interpolation(x_interpol, y_interpol, xp) for xp in x_novos])

# --- Gráfico do Método de Lagrange ---
plt.figure(figsize=(10, 6))
plt.plot(xi, yi, 'o', label='Dados Originais', alpha=0.5)
plt.plot(x_interpol, y_interpol, 'ks', markersize=8, label='Pontos Selecionados')
plt.plot(x_novos, y_lagrange, 'b--', label='Interpolação Lagrange')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas/Receita')
plt.title('Interpolação de Lagrange')
plt.legend()
plt.grid(True)
plt.show()