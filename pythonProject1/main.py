import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROFILE1 = "tczew_starogard"
PROFILE2 = "chelm"
PROFILE3 = "ostrowa"

def load_profile(filename):
    try:
        df = pd.read_csv(filename + ".txt")
        if 'Distance (m)' in df.columns and 'Elevation (m)' in df.columns:
            return df['Distance (m)'].values, df['Elevation (m)'].values
    except pd.errors.ParserError:
        pass

    try:
        df = pd.read_csv(filename + ".txt", header=None)
        if df.shape[1] == 1:
            df = df[0].str.split(expand=True)
        return df[0].astype(float).values, df[1].astype(float).values
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        raise

def generate_linspace(start, stop, num_points):
    return np.linspace(start, stop, num_points)

def interpolate_lagrange(X, Y, num_points=9, num_evaluated=1000, indices=None):
    if indices is None:
        indices = [int(i) for i in generate_linspace(0, len(X) - 1, num_points)]

    def L(x):
        return sum(Y[i] * np.prod([(x - X[j]) / (X[i] - X[j]) for j in indices if i != j], axis=0) for i in indices)

    X_interp = np.linspace(X[0], X[-1], num_evaluated)
    Y_interp = L(X_interp)
    return X_interp, Y_interp, indices

def interpolate_splines(X, Y, num_points=15, num_evaluated=1000, indices=None):
    if indices is None:
        indices = [int(i) for i in generate_linspace(0, len(X) - 1, num_points)]

    n = len(indices)
    h = np.diff(X[indices])
    alpha = np.zeros(n)

    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (Y[indices[i + 1]] - Y[indices[i]]) - (3 / h[i - 1]) * (Y[indices[i]] - Y[indices[i - 1]])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2 * (X[indices[i + 1]] - X[indices[i - 1]]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b, c, d = np.zeros(n - 1), np.zeros(n), np.zeros(n - 1)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (Y[indices[j + 1]] - Y[indices[j]]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    def spline_function(x):
        for j in range(n - 1):
            if X[indices[j]] <= x <= X[indices[j + 1]]:
                dx = x - X[indices[j]]
                return Y[indices[j]] + b[j] * dx + c[j] * dx ** 2 + d[j] * dx ** 3
        return None

    X_interp = np.linspace(X[0], X[-1], num_evaluated)
    Y_interp = np.array([spline_function(x) for x in X_interp])
    return X_interp, Y_interp, indices

def plot_evenly_spaced_with_title(filename, title, num_points=(5, 10, 15, 20), interp_method=interpolate_lagrange, max_y_scale=None):
    X, Y = load_profile(filename)
    fig, axs = plt.subplots(len(num_points), figsize=(10, 20))
    fig.suptitle(title)

    min_y, max_y = min(Y), max(Y)
    if max_y_scale is None:
        max_y_scale = max_y + 10
    for ax, points in zip(axs, num_points):
        X_interp, Y_interp, indices = interp_method(X, Y, num_points=points)
        ax.plot(X, Y, 'b-', label='Dane oryginalne')
        ax.plot(X_interp, Y_interp, 'r--', label='Interpolacja')
        ax.scatter(X[indices], Y[indices], c='g', label='Punkty węzłowe')
        ax.set_xlabel('Odległość (m)')
        ax.set_ylabel('Wysokość (m)')
        ax.set_ylim(min_y, max_y_scale)
        ax.set_title(f'{title}: {points} punktów węzłowych')
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_evenly_spaced_with_title_splines(filename, title, num_points=(5, 10, 15, 20), interp_method=interpolate_splines, max_y_scale=None):
    X, Y = load_profile(filename)
    fig, axs = plt.subplots(len(num_points), figsize=(10, 20))
    fig.suptitle(title)

    min_y, max_y = min(Y), max(Y)
    if max_y_scale is None:
        max_y_scale = max_y + 10  # add a buffer to max_y for better visualization
    for ax, points in zip(axs, num_points):
        X_interp, Y_interp, indices = interp_method(X, Y, num_points=points)
        ax.plot(X, Y, 'b-', label='Dane oryginalne')
        ax.plot(X_interp, Y_interp, 'r--', label='Interpolacja')
        ax.scatter(X[indices], Y[indices], c='g', label='Punkty węzłowe')
        ax.set_xlabel('Odległość (m)')
        ax.set_ylabel('Wysokość (m)')
        ax.set_ylim(min_y, max_y_scale)
        ax.set_title(f'{title}: {points} punktów węzłowych')
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_custom_indices_with_title(filename, title, indices, interp_method=interpolate_lagrange, max_y_scale=None):
    X, Y = load_profile(filename)
    X_interp, Y_interp, _ = interp_method(X, Y, indices=indices)

    min_y, max_y = min(Y), max(Y)
    if max_y_scale is None:
        max_y_scale = max_y + 10

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(X, Y, 'b-', label='Dane oryginalne')
    ax.plot(X_interp, Y_interp, 'r--', label='Interpolacja')
    ax.scatter(X[indices], Y[indices], c='g', label='Punkty węzłowe')
    ax.set_xlabel('Odległość (m)')
    ax.set_ylabel('Wysokość (m)')
    ax.set_ylim(min_y, max_y_scale)
    ax.set_title(f'{title} z nieregularnymi węzłami')
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


print("Analiza interpolacji wielomianowej pierwszej trasy")
plot_evenly_spaced_with_title(PROFILE1, "Interpolacja wielomianowa: Trasa Tczew-Starogard", interp_method=interpolate_lagrange)

print("Analiza interpolacji wielomianowej drugiej trasy")
plot_evenly_spaced_with_title(PROFILE2, "Interpolacja wielomianowa: Trasa Chełm", interp_method=interpolate_lagrange, max_y_scale=100)

print("Analiza interpolacji wielomianowej trzeciej trasy")
plot_evenly_spaced_with_title(PROFILE3, "Interpolacja wielomianowa: Trasa Ostrowa", interp_method=interpolate_lagrange, max_y_scale=2000)

print("Analiza interpolacji funkcjami sklejanymi pierwszej trasy")
plot_evenly_spaced_with_title_splines(PROFILE1, "Interpolacja funkcjami sklejanymi: Trasa Tczew-Starogard", interp_method=interpolate_splines)

print("Analiza interpolacji funkcjami sklejanymi drugiej trasy")
plot_evenly_spaced_with_title_splines(PROFILE2, "Interpolacja funkcjami sklejanymi: Trasa Chełm", interp_method=interpolate_splines, max_y_scale=100)

print("Analiza interpolacji funkcjami sklejanymi trzeciej trasy")
plot_evenly_spaced_with_title_splines(PROFILE3, "Interpolacja funkcjami sklejanymi: Trasa Ostrowa", interp_method=interpolate_splines, max_y_scale=2000)

print("Analiza dodatkowa interpolacji")
custom_indices_10 = [0, 30, 60, 100, 140, 180, 220, 260, 300, 350]
custom_indices_20 = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285]

print("Analiza dodatkowa interpolacji wielomianowej: Trasa Tczew-Starogard (10 węzłów)")
plot_custom_indices_with_title(PROFILE1, "Interpolacja wielomianowa: Trasa Tczew-Starogard (10 węzłów)", indices=custom_indices_10, interp_method=interpolate_lagrange)
print("Analiza dodatkowa interpolacji wielomianowej: Trasa Tczew-Starogard (20 węzłów)")
plot_custom_indices_with_title(PROFILE1, "Interpolacja wielomianowa: Trasa Tczew-Starogard (20 węzłów)", indices=custom_indices_20, interp_method=interpolate_lagrange)

print("Analiza dodatkowa interpolacji funkcjami sklejanymi: Trasa Tczew-Starogard (10 węzłów)")
plot_custom_indices_with_title(PROFILE1, "Interpolacja funkcjami sklejanymi: Trasa Tczew-Starogard (10 węzłów)", indices=custom_indices_10, interp_method=interpolate_splines)
print("Analiza dodatkowa interpolacji funkcjami sklejanymi: Trasa Tczew-Starogard (20 węzłów)")
plot_custom_indices_with_title(PROFILE1, "Interpolacja funkcjami sklejanymi: Trasa Tczew-Starogard (20 węzłów)", indices=custom_indices_20, interp_method=interpolate_splines)

