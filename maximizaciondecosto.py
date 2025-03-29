import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Función para calcular determinantes y puntos de intersección
def calcular_intersecciones(restricciones):
    intersecciones = []
    n = len(restricciones)
    
    # Iterar sobre todas las combinaciones de restricciones para encontrar intersecciones
    for i in range(n):
        for j in range(i+1, n):
            A = np.array([[restricciones[i][0], restricciones[i][1]],
                          [restricciones[j][0], restricciones[j][1]]])
            B = np.array([restricciones[i][2], restricciones[j][2]])
            det = np.linalg.det(A)
            if det != 0:
                punto = np.linalg.solve(A, B)
                intersecciones.append(punto)
    return intersecciones

# Función para verificar si un punto cumple con todas las restricciones
def es_factible(punto, restricciones, simbolos):
    for (a, b, c), simb in zip(restricciones, simbolos):
        lhs = a * punto[0] + b * punto[1]
        if simb == '<=':
            if not lhs <= c + 1e-5:
                return False
        elif simb == '>=':
            if not lhs >= c - 1e-5:
                return False
        elif simb == '=':
            if not np.isclose(lhs, c):
                return False
    return True

# Función para maximizar la función objetivo
def maximizar(costos, restricciones, simbolos):
    # Convertir el problema de maximización a uno de minimización
    c = [-costos[0], -costos[1]]
    A = []
    b = []
    for (a, b_, c_), simb in zip(restricciones, simbolos):
        if simb == '<=':
            A.append([a, b_])
            b.append(c_)
        elif simb == '>=':
            A.append([-a, -b_])
            b.append(-c_)
        elif simb == '=':
            A.append([a, b_])
            b.append(c_)
            A.append([-a, -b_])
            b.append(-c_)
    # Resolver el problema de programación lineal
    res = linprog(c, A_ub=A, b_ub=b, method='highs')
    return res

# Función para graficar las restricciones y la región factible
def graficar(restricciones, simbolos, puntos_factibles, costos, solucion):
    # Calcular interceptos con los ejes para ajustar límites de la gráfica
    interceptos = []
    for (a, b, c), simb in zip(restricciones, simbolos):
        if a != 0:
            interceptos.append((c / a, 0))
        if b != 0:
            interceptos.append((0, c / b))
    interceptos = np.array(interceptos)
    
    # Filtrar solo interceptos positivos
    interceptos = interceptos[np.all(interceptos >= 0, axis=1)]
    
    # Definir los límites de la gráfica con margen adicional
    max_c1 = max(np.max(puntos_factibles[:,0]), solucion.x[0], interceptos[:,0].max()) * 1.2
    max_c2 = max(np.max(puntos_factibles[:,1]), solucion.x[1], interceptos[:,1].max()) * 1.2
    x = np.linspace(-max_c1 * 0.2, max_c1, 400)  # Ajuste para permitir negativos
    plt.figure(figsize=(10, 8))

    # Variables para manejar el desplazamiento de etiquetas
    offset_x = 0.02 * max_c1
    offset_y = 0.02 * max_c2
    label_offsets_x = {}
    label_offsets_y = {}

    # Graficar cada restricción
    for (a, b, c), simb in zip(restricciones, simbolos):
        if b != 0:
            y = (c - a * x) / b
            plt.plot(x, y, label=f'{a}C₁ + {b}C₂ {simb} {c}', linewidth=2)
            
            # Calcular interceptos con los ejes
            intercepto_c1 = c / a if a != 0 else None
            intercepto_c2 = c / b if b != 0 else None
            
            # Marcar e etiquetar intercepto con C1
            if intercepto_c1 is not None and intercepto_c1 >= 0:
                plt.plot(intercepto_c1, 0, 'ko')  # Punto negro
                # Desplazamiento alternativo para evitar superposición
                if intercepto_c1 in label_offsets_x:
                    label_offsets_x[intercepto_c1] += 1
                else:
                    label_offsets_x[intercepto_c1] = 0
                plt.text(intercepto_c1, -offset_y * (1 + label_offsets_x[intercepto_c1]),
                         f'({intercepto_c1:.1f}, 0)', fontsize=9, ha='center')
            
            # Marcar e etiquetar intercepto con C2
            if intercepto_c2 is not None and intercepto_c2 >= 0:
                plt.plot(0, intercepto_c2, 'ko')  # Punto negro
                # Desplazamiento alternativo para evitar superposición
                if intercepto_c2 in label_offsets_y:
                    label_offsets_y[intercepto_c2] += 1
                else:
                    label_offsets_y[intercepto_c2] = 0
                plt.text(-offset_x * (1 + label_offsets_y[intercepto_c2]), intercepto_c2,
                         f'(0, {intercepto_c2:.1f})', fontsize=9, va='center')
            
            # Rellenar según el símbolo de la restricción
            if simb == '<=':
                plt.fill_between(x, y, max_c2, where=(y >=0), color='gray', alpha=0.1)
            elif simb == '>=':
                plt.fill_between(x, y, 0, where=(y >=0), color='gray', alpha=0.1)
        else:
            # Manejar restricciones verticales (a != 0, b = 0)
            x_val = c / a if a != 0 else 0
            plt.axvline(x=x_val, label=f'{a}C₁ {simb} {c}', linewidth=2)
            if x_val >= 0:
                plt.plot(x_val, 0, 'ko')  # Punto negro en C1
                # Desplazamiento alternativo para evitar superposición
                if x_val in label_offsets_x:
                    label_offsets_x[x_val] += 1
                else:
                    label_offsets_x[x_val] = 0
                plt.text(x_val, -offset_y * (1 + label_offsets_x[x_val]),
                         f'({x_val:.1f}, 0)', fontsize=9, ha='center')


    # Graficar la nueva restricción
    a_nueva = 8.88
    b_nueva = -2.25
    c_nueva = 155
    y_nueva = (c_nueva - a_nueva * x) / b_nueva
    plt.plot(x, y_nueva, 'b', label=f'{a_nueva}C₁ - {b_nueva}C₂ <= {c_nueva}', linewidth=2)

    # Calcular el punto óptimo
    AC1 = (80 * (-2.25)) - (155 * 1)
    AC2 = (1 * 155) - (8.88 * 80)
    AT = (1 * (-2.25)) - (8.88 * 1)
    punto_optimo = (AC1 / AT, AC2 / AT)

    plt.plot(punto_optimo[0], punto_optimo[1], 'bs', markersize=10, label='Óptimo', markeredgecolor='white')
    plt.annotate(
        f'Óptimo ({punto_optimo[0]:.2f}, {punto_optimo[1]:.2f})',
        xy=punto_optimo,
        xytext=(punto_optimo[0] + max_c1 * 0.05, punto_optimo[1] + max_c2 * 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white", alpha=0.6)
    )

    # Definir los límites de la gráfica
    plt.xlim(-max_c1 * 0.2, max_c1)
    plt.ylim(0, max_c2)
    plt.xlabel('Estado de Parámetros', fontsize=12)  # Cambiado
    plt.ylabel('Proyección', fontsize=12)
    plt.title('Gráfica de la Función Objetivo y Restricciones', fontsize=14)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

# Función para calcular el resultado y graficar
def calcular():
    try:
        # Obtener coeficientes de la función objetivo
        c1 = float(entry_c1.get())
        c2 = float(entry_c2.get())
        costos = [c1, c2]

        # Obtener restricciones
        restricciones = []
        simbolos = []
        for i in range(3):
            a = float(entries_a[i].get())
            b = float(entries_b[i].get())
            c = float(entries_c[i].get())
            simbolo = símbolos[i].get()
            restricciones.append((a, b, c))
            simbolos.append(simbolo)

        # Agregar la nueva restricción
        restricciones.append((8.88, -2.25, 155))
        simbolos.append('<=')

        # Calcular los puntos factibles
        puntos_factibles = []
        intersecciones = calcular_intersecciones(restricciones)

        # Filtrar puntos factibles
        for punto in intersecciones:
            if es_factible(punto, restricciones, simbolos):
                puntos_factibles.append(punto)

        puntos_factibles = np.array(puntos_factibles)

        # Calcular la solución óptima
        solucion = maximizar(costos, restricciones, simbolos)

        if solucion.success:
            print("Solución óptima encontrada:")
            print(f"C₁ = {solucion.x[0]:.2f}")
            print(f"C₂ = {solucion.x[1]:.2f}")
            print(f"Valor Máximo de la Función Objetivo = {costos[0]*solucion.x[0] + costos[1]*solucion.x[1]:.2f}")
        else:
            messagebox.showerror("Error", "No se pudo encontrar una solución óptima.")
            return

        # Graficar los resultados
        graficar(restricciones, simbolos, puntos_factibles, costos, solucion)

    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores numéricos válidos.")

# Función para limpiar todos los campos
def limpiar():
    entry_c1.delete(0, tk.END)
    entry_c2.delete(0, tk.END)
    for i in range(3):
        entries_a[i].delete(0, tk.END)
        entries_b[i].delete(0, tk.END)
        entries_c[i].delete(0, tk.END)
        símbolos[i].set('<=')  # Restablecer símbolos a '<='

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Maximización de Costos")

# Función Objetivo
frame_objetivo = tk.Frame(root)
frame_objetivo.pack(pady=10)

tk.Label(frame_objetivo, text="Función Objetivo:").grid(row=0, column=0, columnspan=2)

tk.Label(frame_objetivo, text="60 C₁ + 56 C₂").grid(row=1, column=0, columnspan=2)

# Coeficientes de la Función Objetivo
frame_coef = tk.Frame(root)
frame_coef.pack(pady=10)

tk.Label(frame_coef, text="Coeficiente C₁:").grid(row=0, column=0)
entry_c1 = tk.Entry(frame_coef, width=10)
entry_c1.grid(row=0, column=1)
entry_c1.insert(0, "60")

tk.Label(frame_coef, text="Coeficiente C₂:").grid(row=0, column=2)
entry_c2 = tk.Entry(frame_coef, width=10)
entry_c2.grid(row=0, column=3)
entry_c2.insert(0, "56")

# Restricciones
frame_restricciones = tk.Frame(root)
frame_restricciones.pack(pady=10)

tk.Label(frame_restricciones, text="Restricciones:").grid(row=0, column=0, columnspan=7)

entries_a = []
entries_b = []
entries_c = []
símbolos = []

for i in range(3):
    tk.Label(frame_restricciones, text=f"R{i+1}:").grid(row=i+1, column=0)
    a = tk.Entry(frame_restricciones, width=5)
    a.grid(row=i+1, column=1)
    a.insert(0, "1" if i == 0 else "3" if i == 1 else "2")
    entries_a.append(a)

    tk.Label(frame_restricciones, text="C₁ +").grid(row=i+1, column=2)

    b = tk.Entry(frame_restricciones, width=5)
    b.grid(row=i+1, column=3)
    b.insert(0, "1" if i == 0 else "2" if i == 1 else "3")
    entries_b.append(b)

    # Símbolo de la restricción
    simb = tk.StringVar()
    simb.set("<=")
    drop = tk.OptionMenu(frame_restricciones, simb, "<=", ">=", "=")
    drop.grid(row=i+1, column=4)
    símbolos.append(simb)

    tk.Label(frame_restricciones, text="=").grid(row=i+1, column=5)

    c = tk.Entry(frame_restricciones, width=10)
    c.grid(row=i+1, column=6)
    c.insert(0, "80" if i == 0 else "220" if i == 1 else "210")
    entries_c.append(c)

# Botones
frame_botones = tk.Frame(root)
frame_botones.pack(pady=10)

btn_calcular = tk.Button(frame_botones, text="Calcular", command=calcular)
btn_calcular.grid(row=0, column=0, padx=10)

btn_limpiar = tk.Button(frame_botones, text="Limpiar", command=limpiar)
btn_limpiar.grid(row=0, column=1, padx=10)

# Iniciar la interfaz gráfica
root.mainloop()
