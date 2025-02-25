import numpy as np
from sympy.abc import x,y

class MMQ():
    def __init__(self, xi, yi) -> None:
        self.xi = xi
        self.yi = yi 
        self.a, self.b = self.get_a_and_b(xi, yi)

    def _calculate_a(self,xi,yi):
        n = len(xi)
        return (n*np.sum(xi*yi)-np.sum(xi)*np.sum(yi))\
            /(n*np.sum(xi**2)-np.sum(xi)**2)


    def _calculate_b(self,xi,yi,a1):
        n = len(xi)
        return (np.sum(yi) / n) - a1 * (np.sum(xi) / n)

    def get_a_and_b(self, xi, yi):
        a= self._calculate_a(xi,yi)
        b= self._calculate_b(xi,yi,a)

        return a, b 
    
    def get_mmq_linear(self):
        y = self.b + self.a * x 
        return y
    
    def get_mmq_quadratic(self):
        y = self.b + self.a * x **2
        return y 

def main():
    from src.utils.helpers import load_data
    df = load_data("data/ice_cream.csv")

    # Sorting the DataFrame by the column "Temperature"
    df_sorted = df.sort_values(by='Temperature')

    # Extracting xi and yi from df_sorted
    xi = df_sorted['Temperature'].values
    yi = df_sorted['Revenue'].values


    mmq = MMQ(xi, yi)

    y_linear = mmq.get_mmq_linear()

    y_quad  = mmq.get_mmq_quadratic()
    
    print(f"Linear MMQ Result: {y_linear}, Quadratic MMQ Result: {y_quad}")

if __name__ == "__main__":
    main()