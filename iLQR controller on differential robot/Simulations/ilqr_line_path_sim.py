import numpy as np
import matplotlib.pyplot as plt

# ====================
# Parámetros del modelo
# ====================
m = 0.825     # masa [kg]
J = 0.05      # inercia [kg*m^2]
r = 0.0025    # radio de la rueda [m]
L = 0.018     # distancia entre ruedas [m]
G = 4.4       # relación de reducción (gear ratio)
eta = 0.9     # eficiencia mecánica (0 < eta <= 1)
dt = 0.1      # paso de tiempo [s]
N = 50        # pasos de predicción
max_tau = 1.0 # saturación de torque

# ====================
# Matrices de costo
# ====================
Q = np.diag([10, 10, 10, 1, 1])
Rmat = 0.01 * np.eye(2)
Qf = 50 * np.eye(5)

# ====================
# Función para envolver ángulos
# ====================
def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ====================
# Estado inicial y referencia
# ====================
x0 = np.array([0, 0, 0, 0, 0], dtype=np.float64)
x_ref = np.zeros((5, N))
for k in range(N):
    x_ref[:, k] = [0.2 * k * dt, 0, 0, 0.2, 0]

# ====================
# Inicialización
# ====================
x = np.zeros((5, N))
x[:, 0] = x0
u = np.zeros((2, N - 1))

def dynamics(x, u, m, J, r, L, G, eta, dt):
    """
    x: [x, y, theta, v, omega]
    u: [tau_md, tau_mi] - torques en el motor
    m: masa
    J: momento de inercia
    r: radio rueda
    L: distancia entre ruedas
    G: relacion de engranes (gear ratio)
    eta: eficiencia mecánica (0 < eta <= 1)
    dt: paso de tiempo
    """
    x_pos, y_pos, theta, v, omega = x
    tau_md, tau_mi = u

    # Torque en ruedas, considerando eficiencia y relación de engranes
    tau_d = eta * G * tau_md
    tau_i = eta * G * tau_mi

    # Derivadas
    v_dot = (r / m) * (tau_d + tau_i)
    omega_dot = (r / (J * L)) * (tau_d - tau_i)

    dx = np.array([
        v * np.cos(theta),  # dx/dt
        v * np.sin(theta),  # dy/dt
        omega,              # dtheta/dt
        v_dot,              # dv/dt
        omega_dot           # domega/dt
    ])

    return x + dt * dx


def linearize(x, u, m, J, r, G, eta, L, dt):
    theta, v = x[2], x[3]
    A = np.eye(5)
    A[0, 2] = -dt * v * np.sin(theta)
    A[0, 3] = dt * np.cos(theta)
    A[1, 2] = dt * v * np.cos(theta)
    A[1, 3] = dt * np.sin(theta)
    A[2, 4] = dt

    B = np.zeros((5, 2))
    # Incorporamos G y eta en B
    B[3, :] = dt * np.array([r / m, r / m]) * eta * G
    B[4, :] = dt * np.array([r / (J * L), -r / (J * L)]) * eta * G
    return A, B

# ====================
# iLQR MAIN LOOP
# ====================
max_iter = 20
costs = []

for it in range(max_iter):
    # Forward pass
    for k in range(N - 1):
        x[:, k + 1] = dynamics(x[:, k], u[:, k], m, J, r, L, G, eta, dt)

    # Backward pass
    Vx = Qf @ (x[:, -1] - x_ref[:, -1])
    Vx[2] = angle_wrap(Vx[2])
    Vxx = Qf
    K = np.zeros((2, 5, N - 1))
    d = np.zeros((2, N - 1))

    for k in reversed(range(N - 1)):
        A, B = linearize(x[:, k], u[:, k], m, J, r, G, eta, L, dt)
        dx_err = x[:, k] - x_ref[:, k]
        dx_err[2] = angle_wrap(dx_err[2])

        Qx = Q @ dx_err + A.T @ Vx
        Qu = Rmat @ u[:, k] + B.T @ Vx
        Qxx = Q + A.T @ Vxx @ A
        Quu = Rmat + B.T @ Vxx @ B
        Qux = B.T @ Vxx @ A

        inv_Quu = np.linalg.pinv(Quu)
        K[:, :, k] = -inv_Quu @ Qux
        d[:, k] = -inv_Quu @ Qu

        Vx = Qx + K[:, :, k].T @ Quu @ d[:, k] + K[:, :, k].T @ Qu + Qux.T @ d[:, k]
        Vxx = Qxx + K[:, :, k].T @ Quu @ K[:, :, k] + K[:, :, k].T @ Qux + Qux.T @ K[:, :, k]

    # Forward pass con nueva política
    x_new = np.zeros((5, N))
    u_new = np.zeros((2, N - 1))
    x_new[:, 0] = x0

    for k in range(N - 1):
        dx = x_new[:, k] - x[:, k]  # error respecto a trayectoria previa
        dx[2] = angle_wrap(dx[2])   # envolver ángulo
        u_new[:, k] = u[:, k] + d[:, k] + K[:, :, k] @ dx
        u_new[:, k] = np.clip(u_new[:, k], -max_tau, max_tau)
        x_new[:, k + 1] = dynamics(x_new[:, k], u_new[:, k], m, J, r, L, G, eta, dt)

    # Costo total
    cost = 0
    for k in range(N - 1):
        dx = x_new[:, k] - x_ref[:, k]
        dx[2] = angle_wrap(dx[2])
        cost += dx @ Q @ dx + u_new[:, k] @ Rmat @ u_new[:, k]
    dx_final = x_new[:, -1] - x_ref[:, -1]
    dx_final[2] = angle_wrap(dx_final[2])
    cost += dx_final @ Qf @ dx_final
    costs.append(cost)

    # Actualización
    x = x_new
    u = u_new

    print(f"Iteración {it + 1}: Costo = {cost:.4f}")

# ====================
# Graficar resultados
# ====================
plt.figure(figsize=(8, 5))
plt.plot(x[0, :], x[1, :], label="Trayectoria seguida", linewidth=2)
plt.plot(x_ref[0, :], x_ref[1, :], '--', label="Trayectoria deseada", linewidth=2)
plt.quiver(x[0, ::2], x[1, ::2], np.cos(x[2, ::2]), np.sin(x[2, ::2]), scale=10, color='gray')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Trayectoria del robot")
plt.grid()
plt.axis("equal")
plt.legend()

plt.figure(figsize=(8, 4))
plt.plot(costs, marker='o')
plt.title("Costo total por iteración")
plt.xlabel("Iteración")
plt.ylabel("Costo")
plt.grid()

plt.figure(figsize=(8, 4))
plt.plot(u[0, :], label=r"$\tau_{md}$")
plt.plot(u[1, :], label=r"$\tau_{mi}$")
plt.xlabel("Paso de tiempo")
plt.ylabel("Torque [Nm]")
plt.title("Torques aplicados en el motor")
plt.legend()
plt.grid()

plt.show()
