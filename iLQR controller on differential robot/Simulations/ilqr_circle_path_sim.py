import numpy as np
import matplotlib.pyplot as plt

# Parámetros del carrito
m = 0.50
J = 0.0005
r = 0.05
L = 0.2
R = 1.0
dt = 0.1
N = 50

# Matrices de costo
Q = np.diag([500, 500, 500, 1, 1])
Rmat = 70 * np.eye(2)
Qf = np.diag([100, 100, 200, 10, 10])

# Trayectoria deseada: medio círculo antihorario
r_traj = 1.0
theta_traj = np.linspace(np.pi/2, 3*np.pi/2, N)

x_circ = r_traj * np.cos(theta_traj)
y_circ = r_traj * np.sin(theta_traj)

dx = np.gradient(x_circ, dt)
dy = np.gradient(y_circ, dt)
theta_tangent = np.arctan2(dy, dx)

ds = np.sqrt(dx**2 + dy**2)
v_ref = ds / dt
omega_ref = np.gradient(theta_tangent, dt)

x_ref = np.zeros((5, N))
x_ref[0, :] = x_circ
x_ref[1, :] = y_circ
x_ref[2, :] = theta_tangent
x_ref[3, :] = v_ref
x_ref[4, :] = omega_ref

# Estado inicial
x0 = np.array([x_circ[0], y_circ[0], theta_tangent[0], 0.0, 0.0])

# Inicialización
x = np.zeros((5, N))
x[:, 0] = x0
u = np.zeros((2, N - 1))

def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def dynamics(x, u, m, J, r, R, L, dt):
    theta, v, omega = x[2], x[3], x[4]
    tau_d, tau_i = u[0], u[1]
    v_dot = (r / (m * R)) * (tau_d + tau_i)
    omega_dot = (r / (J * R * L)) * (tau_d - tau_i)
    dx = np.array([
        v * np.cos(theta),
        v * np.sin(theta),
        omega,
        v_dot,
        omega_dot
    ])
    return x + dt * dx

def linearize(x, u, m, J, r, R, L, dt):
    theta, v = x[2], x[3]
    A = np.eye(5)
    A[0, 2] = -dt * v * np.sin(theta)
    A[0, 3] = dt * np.cos(theta)
    A[1, 2] = dt * v * np.cos(theta)
    A[1, 3] = dt * np.sin(theta)
    A[2, 4] = dt
    B = np.zeros((5, 2))
    B[3, :] = dt * np.array([r / (m * R), r / (m * R)])
    B[4, :] = dt * np.array([r / (J * R * L), -r / (J * R * L)])
    return A, B

max_iter = 20
costs = []

for it in range(max_iter):
    # Forward pass con control actual
    for k in range(N - 1):
        x[:, k + 1] = dynamics(x[:, k], u[:, k], m, J, r, R, L, dt)

    # Backward pass
    Vx = Qf @ (x[:, -1] - x_ref[:, -1])
    Vx[2] = angle_wrap(Vx[2])
    Vxx = Qf.copy()
    K = np.zeros((2, 5, N - 1))
    d = np.zeros((2, N - 1))

    for k in reversed(range(N - 1)):
        A, B = linearize(x[:, k], u[:, k], m, J, r, R, L, dt)
        dxk = x[:, k] - x_ref[:, k]
        dxk[2] = angle_wrap(dxk[2])

        Qx = Q @ dxk + A.T @ Vx
        Qu = Rmat @ u[:, k] + B.T @ Vx
        Qxx = Q + A.T @ Vxx @ A
        Quu = Rmat + B.T @ Vxx @ B
        Qux = B.T @ Vxx @ A

        inv_Quu = np.linalg.inv(Quu)
        K[:, :, k] = -inv_Quu @ Qux
        d[:, k] = -inv_Quu @ Qu

        Vx = Qx + K[:, :, k].T @ Quu @ d[:, k] + K[:, :, k].T @ Qu + Qux.T @ d[:, k]
        Vxx = Qxx + K[:, :, k].T @ Quu @ K[:, :, k] + K[:, :, k].T @ Qux + Qux.T @ K[:, :, k]

    # Forward pass con nueva política
    x_new = np.zeros((5, N))
    u_new = np.zeros((2, N - 1))
    x_new[:, 0] = x0

    for k in range(N - 1):
        dxk = x_new[:, k] - x[:, k]   # <-- clave: usar x y no x_ref
        dxk[2] = angle_wrap(dxk[2])
        u_new[:, k] = u[:, k] + d[:, k] + K[:, :, k] @ dxk
        x_new[:, k + 1] = dynamics(x_new[:, k], u_new[:, k], m, J, r, R, L, dt)

    # Actualizar para siguiente iteración
    x = x_new
    u = u_new

    # Costo para monitorear convergencia
    cost = 0
    for k in range(N - 1):
        dxk = x[:, k] - x_ref[:, k]
        dxk[2] = angle_wrap(dxk[2])
        cost += dxk @ Q @ dxk + u[:, k] @ Rmat @ u[:, k]
    dxk = x[:, -1] - x_ref[:, -1]
    dxk[2] = angle_wrap(dxk[2])
    cost += dxk @ Qf @ dxk
    costs.append(cost)
    print(f"Iteración {it + 1}: Costo = {cost:.4f}")

# Graficar
plt.figure(figsize=(8,5))
plt.plot(x[0, :], x[1, :], label="Trayectoria seguida", linewidth=2)
plt.plot(x_ref[0, :], x_ref[1, :], '--', label="Trayectoria deseada", linewidth=2, color='orange')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Seguimiento de trayectoria circular con iLQR")
plt.axis("equal")
plt.grid()
plt.legend()

plt.figure()
plt.plot(costs, marker='o')
plt.xlabel("Iteración")
plt.ylabel("Costo total")
plt.title("Convergencia del iLQR")
plt.grid()

plt.figure()
plt.plot(u[0, :], label=r"$\tau_d$")
plt.plot(u[1, :], label=r"$\tau_i$")
plt.xlabel("Paso de tiempo")
plt.ylabel("Torque [Nm]")
plt.title("Torques aplicados")
plt.legend()
plt.grid()

plt.show()
