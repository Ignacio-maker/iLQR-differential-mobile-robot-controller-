import numpy as np
import Jetson.GPIO as GPIO
import time
import os
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt

# ====================
# PARÁMETROS DEL ROBOT
# ====================
m = 1.2     # masa [kg]
J = 0.005   # inercia [kg*m^2]
r = 0.025   # radio de la rueda [m]
L = 0.18    # distancia entre ruedas [m]
G = 4.4     # relación de engranes
eta = 0.9   # eficiencia mecánica

# ====================
# PWM
# ====================
PWM_CHIP = "/sys/class/pwm/pwmchip0"
PWM_DER = f"{PWM_CHIP}/pwm0"
PWM_IZQ = f"{PWM_CHIP}/pwm2"

# Pines dirección
IN1 = 23
IN2 = 24
IN3 = 21
IN4 = 22

# Pines de encoder
ENC_IZQ_A = 15
ENC_IZQ_B = 13
ENC_DER_A = 18
ENC_DER_B = 16

GPIO.setmode(GPIO.BOARD)
GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT, initial=GPIO.LOW)
GPIO.setup([ENC_IZQ_A, ENC_IZQ_B, ENC_DER_A, ENC_DER_B], GPIO.IN)

# ====================
# ENCODER - Variables globales
# ====================
contador_der = 0
contador_izq = 0
last_state_der = GPIO.input(ENC_DER_A)
last_state_izq = GPIO.input(ENC_IZQ_A)

# ====================
# CALLBACKS DE ENCODER
# ====================
def update_encoder_der(channel):
    global contador_der, last_state_der
    estado_actual = GPIO.input(ENC_DER_A)
    if estado_actual != last_state_der:
        if GPIO.input(ENC_DER_B) != estado_actual:
            contador_der += 1
        else:
            contador_der -= 1
    last_state_der = estado_actual

def update_encoder_izq(channel):
    global contador_izq, last_state_izq
    estado_actual = GPIO.input(ENC_IZQ_A)
    if estado_actual != last_state_izq:
        if GPIO.input(ENC_IZQ_B) != estado_actual:
            contador_izq += 1
        else:
            contador_izq -= 1
    last_state_izq = estado_actual

GPIO.add_event_detect(ENC_DER_A, GPIO.BOTH, callback=update_encoder_der)
GPIO.add_event_detect(ENC_IZQ_A, GPIO.BOTH, callback=update_encoder_izq)

# ====================
# FUNCIONES PWM por sysfs
# ====================
def export_pwm(path):
    if not os.path.exists(path):
        pwm_num = path[-1]
        with open(f"{PWM_CHIP}/export", 'w') as f:
            f.write(pwm_num)
        time.sleep(0.1)

def enable_pwm(path, freq=100):
    period_ns = int(1e9 / freq)
    with open(f"{path}/period", 'w') as f:
        f.write(str(period_ns))
    with open(f"{path}/duty_cycle", 'w') as f:
        f.write("0")
    with open(f"{path}/enable", 'w') as f:
        f.write("1")
    return period_ns

def set_duty(path, duty_percent, period_ns):
    duty_ns = int((duty_percent / 100.0) * period_ns)
    with open(f"{path}/duty_cycle", 'w') as f:
        f.write(str(duty_ns))

def stop_pwm(path):
    if os.path.exists(path):
        with open(f"{path}/enable", 'w') as f:
            f.write("0")

def set_motor_pwm_dir_sysfs(pwm_path, dir_pin1, dir_pin2, pwm_value, period_ns):
    if pwm_value > 0.0:
        GPIO.output(dir_pin1, GPIO.HIGH)
        GPIO.output(dir_pin2, GPIO.LOW)
        duty = min(pwm_value, 100)
    elif pwm_value < 0.0:
        GPIO.output(dir_pin1, GPIO.LOW)
        GPIO.output(dir_pin2, GPIO.HIGH)
        duty = min(abs(pwm_value), 100)
    else:
        GPIO.output(dir_pin1, GPIO.LOW)
        GPIO.output(dir_pin2, GPIO.LOW)
        duty = 0
    if duty >= 0:
        set_duty(pwm_path, duty, period_ns)

# ====================
# POLINOMIO PWM -> TORQUE
# ====================
PWM_vals = np.array([20, 40, 60, 80, 100])
torques = np.array([0.014, 0.036, 0.057, 0.078, 0.0980665])
torque_to_pwm = np.poly1d(np.polyfit(torques, PWM_vals, deg=2))

# ====================
# FUNCIÓN DE ODOMETRÍA
# ====================
PULSOS_POR_REV = 360
DISTANCIA_POR_PULSO = 2 * np.pi * r / PULSOS_POR_REV

x_global = 0.0
y_global = 0.0
theta_global = 0.0
contador_der_anterior = 0
contador_izq_anterior = 0

def leer_odometria():
    global contador_der_anterior, contador_izq_anterior
    global x_global, y_global, theta_global
    d_der = (contador_der - contador_der_anterior) * DISTANCIA_POR_PULSO
    d_izq = (contador_izq - contador_izq_anterior) * DISTANCIA_POR_PULSO
    contador_der_anterior = contador_der
    contador_izq_anterior = contador_izq
    d = (d_der + d_izq) / 2.0
    d_theta = (d_der - d_izq) / L
    theta_global += d_theta
    x_global += d * np.cos(theta_global)
    y_global += d * np.sin(theta_global)
    v = d / dt
    omega = d_theta / dt
    return np.array([x_global, y_global, theta_global, v, omega])

# ====================
# ILQR
# ====================
def dynamics(x, u):
    theta, v, omega = x[2], x[3], x[4]
    tau_md, tau_mi = u
    tau_d = eta * G * tau_md
    tau_i = eta * G * tau_mi
    v_dot = (r / m) * (tau_d + tau_i)
    omega_dot = (r / (J * L)) * (tau_d - tau_i)
    dx = np.array([
        v * np.cos(theta),
        v * np.sin(theta),
        omega,
        v_dot,
        omega_dot
    ])
    return x + dt * dx

def linearize(x, u):
    theta, v = x[2], x[3]
    A = np.eye(5)
    A[0, 2] = -dt * v * np.sin(theta)
    A[0, 3] = dt * np.cos(theta)
    A[1, 2] = dt * v * np.cos(theta)
    A[1, 3] = dt * np.sin(theta)
    A[2, 4] = dt
    B = np.zeros((5, 2))
    B[3, :] = dt * np.array([r / m, r / m]) * eta * G
    B[4, :] = dt * np.array([r / (J * L), -r / (J * L)]) * eta * G
    return A, B

def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def print_debug_info(u, pwm_der, pwm_izq):
    tau_d = u[0]
    tau_i = u[1]
    print(f"Torque derecho: {tau_d:.5f} Nm, Torque izquierdo: {tau_i:.5f} Nm")
    print(f"PWM derecho: {pwm_der:.2f} %, PWM izquierdo: {pwm_izq:.2f} %")
    print("-" * 40)

def ilqr(x0, x_ref):
    x = np.zeros((5, N))
    x[:, 0] = x0
    u = np.zeros((2, N - 1))
    for it in range(max_iter):
        for k in range(N - 1):
            x[:, k + 1] = dynamics(x[:, k], u[:, k])
        Vx = Qf @ (x[:, -1] - x_ref[:, -1])
        Vx[2] = angle_wrap(Vx[2])
        Vxx = Qf
        K = np.zeros((2, 5, N - 1))
        d = np.zeros((2, N - 1))
        for k in reversed(range(N - 1)):
            A, B = linearize(x[:, k], u[:, k])
            dx = x[:, k] - x_ref[:, k]
            dx[2] = angle_wrap(dx[2])
            Qx = Q @ dx + A.T @ Vx
            Qu = Rmat @ u[:, k] + B.T @ Vx
            Qxx = Q + A.T @ Vxx @ A
            Quu = Rmat + B.T @ Vxx @ B
            Qux = B.T @ Vxx @ A
            inv_Quu = np.linalg.pinv(Quu)
            K[:, :, k] = -inv_Quu @ Qux
            d[:, k] = -inv_Quu @ Qu
            Vx = Qx + K[:, :, k].T @ Quu @ d[:, k] + K[:, :, k].T @ Qu + Qux.T @ d[:, k]
            Vxx = Qxx + K[:, :, k].T @ Quu @ K[:, :, k] + K[:, :, k].T @ Qux + Qux.T @ K[:, :, k]
    return u[:, 0]

# ====================
# ILQR PARAMS & NUEVA TRAYECTORIA REFERENCIA
# ====================
dt = 0.1
N = 50
max_iter = 20
Q = np.diag([20, 20, 20, 1, 1])
Qf = 50 * np.eye(5)
Rmat = 0.01 * np.eye(2)

# Nueva trayectoria de referencia ajustada a la curva de la imagen
x_ref = np.linspace(0, -0.025, N)  # X de 0 a -0.025
y_ref = -0.15 * (x_ref / -0.025)**3 - 0.15  # Curva cúbica descendente más pronunciada

# Ajustar para que termine en (0, -0.15)
y_ref[-1] = -0.15

# Calcular derivadas para velocidades
dx = np.gradient(x_ref, dt)
dy = np.gradient(y_ref, dt)
theta_tangent = np.arctan2(dy, dx)
ds = np.sqrt(dx**2 + dy**2)
v_ref = ds / dt
omega_ref = np.gradient(theta_tangent, dt)

x_ref_global = np.zeros((5, N))
x_ref_global[0, :] = x_ref
x_ref_global[1, :] = y_ref
x_ref_global[2, :] = theta_tangent
x_ref_global[3, :] = v_ref
x_ref_global[4, :] = omega_ref

# ====================
# INICIA PWM
# ====================
export_pwm(PWM_DER)
export_pwm(PWM_IZQ)
period_ns = enable_pwm(PWM_DER)
enable_pwm(PWM_IZQ)

# ===================================
# ALMACENAMIENTO PARA GRÁFICAS
# ===================================
x_odom = []
y_odom = []
posiciones = []  # Almacenará tuplas (x, y, tiempo)

# ====================
# BUCLE DE CONTROL
# ====================
print("Iniciando control...")
print("Posiciones en tiempo real (x, y):")
print("---------------------------------")

try:
    iter_count = 0
    start_time = time.time()
    
    while True:
        x = leer_odometria()
        x_odom.append(x[0])
        y_odom.append(x[1])
        current_time = time.time() - start_time
        posiciones.append((x[0], x[1], current_time))
        
        # Imprimir posición cada iteración
        print(f"x = {x[0]:.6f}, y = {x[1]:.6f}, t = {current_time:.2f}s")
        
        # Solo calcular control si tenemos suficientes puntos de referencia
        if len(x_odom) > 5:
            u = ilqr(x, x_ref_global)
            pwm_der = torque_to_pwm(u[0])
            pwm_izq = torque_to_pwm(u[1])
            pwm_der = np.clip(pwm_der, -100, 100)
            pwm_izq = np.clip(pwm_izq, -100, 100)
            
            PWM_MIN = 25
            if 0 < abs(pwm_der) < PWM_MIN:
                pwm_der = PWM_MIN * np.sign(pwm_der)
            if 0 < abs(pwm_izq) < PWM_MIN:
                pwm_izq = 60 * np.sign(pwm_izq)

            set_motor_pwm_dir_sysfs(PWM_DER, IN1, IN2, pwm_der, period_ns)
            set_motor_pwm_dir_sysfs(PWM_IZQ, IN4, IN3, pwm_izq, period_ns)
        else:
            # Mover lentamente al inicio para recolectar datos
            set_motor_pwm_dir_sysfs(PWM_DER, IN1, IN2, 20, period_ns)
            set_motor_pwm_dir_sysfs(PWM_IZQ, IN4, IN3, 20, period_ns)
        
        time.sleep(dt)
        iter_count += 1

except KeyboardInterrupt:
    print("\n[INFO] Programa terminado por el usuario")
    set_duty(PWM_DER, 0, period_ns)
    set_duty(PWM_IZQ, 0, period_ns)
    stop_pwm(PWM_DER)
    stop_pwm(PWM_IZQ)
    GPIO.cleanup()
    
    # ===================================
    # GENERACIÓN DE GRÁFICAS - SOLUCIÓN MEJORADA
    # ===================================
    if len(x_odom) > 1:  # Solo generar gráficas si hay datos
        # Crear figura con 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Gráfica 1: Trayectoria de referencia
        ax1.plot(x_ref, y_ref, 'r-', linewidth=2)
        ax1.plot(x_ref[0], y_ref[0], 'go', markersize=8, label='Inicio')
        ax1.plot(x_ref[-1], y_ref[-1], 'ro', markersize=8, label='Fin')
        ax1.set_title('Trayectoria de Referencia', fontsize=14)
        ax1.set_xlabel('Posición X (m)', fontsize=12)
        ax1.set_ylabel('Posición Y (m)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True)
        ax1.axis('equal')
        
        # Gráfica 2: Trayectoria reconstruida por odometría
        ax2.plot(x_odom, y_odom, 'b-', linewidth=2)
        ax2.plot(x_odom[0], y_odom[0], 'go', markersize=8, label='Inicio')
        ax2.plot(x_odom[-1], y_odom[-1], 'ro', markersize=8, label='Fin')
        ax2.set_title('Trayectoria Reconstruida por Odometría', fontsize=14)
        ax2.set_xlabel('Posición X (m)', fontsize=12)
        ax2.set_ylabel('Posición Y (m)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True)
        ax2.axis('equal')
        
        # Gráfica 3: Comparación entre trayectorias
        ax3.plot(x_ref, y_ref, 'r--', linewidth=2, label='Referencia')
        ax3.plot(x_odom, y_odom, 'b-', alpha=0.7, label='Odometría')
        ax3.plot(x_ref[0], y_ref[0], 'go', markersize=8, label='Inicio Ref')
        ax3.plot(x_ref[-1], y_ref[-1], 'mo', markersize=8, label='Fin Ref')
        ax3.plot(x_odom[0], y_odom[0], 'co', markersize=8, label='Inicio Odom')
        ax3.plot(x_odom[-1], y_odom[-1], 'yo', markersize=8, label='Fin Odom')
        ax3.set_title('Comparación: Referencia vs Odometría', fontsize=14)
        ax3.set_xlabel('Posición X (m)', fontsize=12)
        ax3.set_ylabel('Posición Y (m)', fontsize=12)
        ax3.legend(loc='upper right')
        ax3.grid(True)
        ax3.axis('equal')
        
        plt.tight_layout()
        
        # Guardar gráficas en el directorio actual
        output_path = 'odometria_comparacion.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Gráficas guardadas en {os.path.abspath(output_path)}")
        
        # Guardar datos en un archivo CSV
        csv_path = 'datos_odometria.csv'
        with open(csv_path, 'w') as f:
            f.write("Tiempo(s),Posicion_X,Posicion_Y\n")
            for pos in posiciones:
                f.write(f"{pos[2]:.2f},{pos[0]:.6f},{pos[1]:.6f}\n")
        print(f"[INFO] Datos guardados en {os.path.abspath(csv_path)}")
    else:
        print("[WARNING] No hay suficientes datos para generar gráficas")
    
    # Imprimir resumen de posiciones
    if x_odom and y_odom:
        print("\nResumen de posiciones registradas:")
        print("---------------------------------")
        print(f"Posición inicial: x={x_odom[0]:.6f}, y={y_odom[0]:.6f}")
        print(f"Posición final:   x={x_odom[-1]:.6f}, y={y_odom[-1]:.6f}")
        print(f"Total de puntos registrados: {len(x_odom)}")