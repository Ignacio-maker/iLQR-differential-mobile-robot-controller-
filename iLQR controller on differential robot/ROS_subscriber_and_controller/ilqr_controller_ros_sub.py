#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import Jetson.GPIO as GPIO
import time
import os
import rospy
from std_msgs.msg import String

# ====================
# PARÁMETROS DEL ROBOT
# ====================
m = 1.2          # masa [kg]
J = 0.005        # inercia [kg*m^2]
r = 0.025        # radio de la rueda [m]
L = 0.18         # distancia entre ruedas [m]
G = 4.4          # relación de engranes
eta = 0.9        # eficiencia mecánica

# ====================
# CONFIGURACIÓN PWM
# ====================
PWM_CHIP = "/sys/class/pwm/pwmchip0"
PWM_DER = PWM_CHIP + "/pwm0"
PWM_IZQ = PWM_CHIP + "/pwm2"

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

# Configuración GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT, initial=GPIO.LOW)
GPIO.setup([ENC_IZQ_A, ENC_IZQ_B, ENC_DER_A, ENC_DER_B], GPIO.IN)

# ====================
# VARIABLES GLOBALES
# ====================
contador_der = 0
contador_izq = 0
last_state_der = GPIO.input(ENC_DER_A)
last_state_izq = GPIO.input(ENC_IZQ_A)
x_global = 0.0
y_global = 0.0
theta_global = 0.0
contador_der_anterior = 0
contador_izq_anterior = 0
trayectoria_activa = None  # 'S': recta, 'L': izquierda, 'R': derecha
period_ns = None

# ====================
# PARÁMETROS ILQR
# ====================
dt = 0.1
N = 50
max_iter = 20
Q = np.diag([20, 20, 20, 1, 1])
Qf = 50 * np.eye(5)
Rmat = 0.01 * np.eye(2)
PULSOS_POR_REV = 360
DISTANCIA_POR_PULSO = 2 * np.pi * r / PULSOS_POR_REV
PWM_MIN = 25

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
# FUNCIONES PWM
# ====================
def export_pwm(path):
    if not os.path.exists(path):
        pwm_num = path[-1]
        with open(PWM_CHIP + "/export", 'w') as f:
            f.write(pwm_num)
        time.sleep(0.1)

def enable_pwm(path, freq=100):
    period_ns = int(1e9 / freq)
    with open(path + "/period", 'w') as f:
        f.write(str(period_ns))
    with open(path + "/duty_cycle", 'w') as f:
        f.write("0")
    with open(path + "/enable", 'w') as f:
        f.write("1")
    return period_ns

def set_duty(path, duty_percent, period_ns):
    duty_ns = int((duty_percent / 100.0) * period_ns)
    with open(path + "/duty_cycle", 'w') as f:
        f.write(str(duty_ns))

def stop_pwm(path):
    if os.path.exists(path):
        with open(path + "/enable", 'w') as f:
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
# MODELO Y CONTROL
# ====================
PWM_vals = np.array([20, 40, 60, 80, 100])
torques = np.array([0.014, 0.036, 0.057, 0.078, 0.0980665])
torque_to_pwm = np.poly1d(np.polyfit(torques, PWM_vals, deg=2))

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

def ilqr(x0, x_ref):
    x = np.zeros((5, N))
    x[:, 0] = x0
    u = np.zeros((2, N - 1))
    for it in range(max_iter):
        for k in range(N - 1):
            x[:, k + 1] = dynamics(x[:, k], u[:, k])
        Vx = np.dot(Qf, (x[:, -1] - x_ref[:, -1]))
        Vx[2] = angle_wrap(Vx[2])
        Vxx = Qf
        K = np.zeros((2, 5, N - 1))
        d = np.zeros((2, N - 1))
        for k in reversed(range(N - 1)):
            A, B = linearize(x[:, k], u[:, k])
            dx = x[:, k] - x_ref[:, k]
            dx[2] = angle_wrap(dx[2])
            Qx = np.dot(Q, dx) + np.dot(A.T, Vx)
            Qu = np.dot(Rmat, u[:, k]) + np.dot(B.T, Vx)
            Qxx = Q + np.dot(np.dot(A.T, Vxx), A)
            Quu = Rmat + np.dot(np.dot(B.T, Vxx), B)
            Qux = np.dot(np.dot(B.T, Vxx), A)
            inv_Quu = np.linalg.pinv(Quu)
            K[:, :, k] = -np.dot(inv_Quu, Qux)
            d[:, k] = -np.dot(inv_Quu, Qu)
            Vx = Qx + np.dot(K[:, :, k].T, np.dot(Quu, d[:, k])) + np.dot(K[:, :, k].T, Qu) + np.dot(Qux.T, d[:, k])
            Vxx = Qxx + np.dot(K[:, :, k].T, np.dot(Quu, K[:, :, k])) + np.dot(K[:, :, k].T, Qux) + np.dot(Qux.T, K[:, :, k])
    return u[:, 0]

# ====================
# TRAYECTORIAS
# ====================
def generar_trayectoria_recta():
    x_ref = np.zeros((5, N))
    for k in range(N):
        x_ref[:, k] = [0.2 * k * dt, 0, 0, 0.2, 0]
    return x_ref

def generar_trayectoria_circular(derecha=True):
    r_traj = 0.1
    if derecha:
        theta_traj = np.linspace(np.pi / 2, 3 * np.pi / 2, N)
    else:
        theta_traj = np.linspace(np.pi / 2, -np.pi / 2, N)
    
    x_circ = r_traj * np.cos(theta_traj)
    y_circ = r_traj * np.sin(theta_traj)
    dx = np.gradient(x_circ, dt)
    dy = np.gradient(y_circ, dt)
    theta_tangent = np.arctan2(dy, dx)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    v_ref = ds / dt
    omega_ref = np.gradient(theta_tangent, dt)
    
    x_ref = np.zeros((5, N))
    x_ref[0, :] = x_circ
    x_ref[1, :] = y_circ
    x_ref[2, :] = theta_tangent
    x_ref[3, :] = v_ref
    x_ref[4, :] = omega_ref
    return x_ref

# ====================
# MANEJO DE COMANDOS ROS
# ====================
def manejar_direccion(msg):
    global trayectoria_activa, x_global, y_global, theta_global
    global contador_der_anterior, contador_izq_anterior
    
    direccion = msg.data
    print("Dirección recibida: " + direccion)
    
    # Detener motores y reiniciar odometría
    set_duty(PWM_DER, 0, period_ns)
    set_duty(PWM_IZQ, 0, period_ns)
    x_global = y_global = theta_global = 0.0
    contador_der_anterior = contador_der
    contador_izq_anterior = contador_izq
    
    if direccion in ['S', 'L', 'R']:
        trayectoria_activa = direccion
    else:
        trayectoria_activa = None
        print("Dirección no reconocida")

# ====================
# BUCLE PRINCIPAL
# ====================
def main():
    global period_ns, trayectoria_activa
    
    # Inicializar PWM
    export_pwm(PWM_DER)
    export_pwm(PWM_IZQ)
    period_ns = enable_pwm(PWM_DER)
    enable_pwm(PWM_IZQ)
    
    # Inicializar nodo ROS
    rospy.init_node('robot_control_unificado')
    rospy.Subscriber('/direccion', String, manejar_direccion)
    print("Nodo ROS iniciado. Esperando comandos...")
    
    try:
        while not rospy.is_shutdown():
            if trayectoria_activa:
                x = leer_odometria()
                
                # Seleccionar trayectoria según comando
                if trayectoria_activa == 'S':
                    x_ref = generar_trayectoria_recta()
                elif trayectoria_activa == 'R':
                    x_ref = generar_trayectoria_circular(derecha=True)
                elif trayectoria_activa == 'L':
                    x_ref = generar_trayectoria_circular(derecha=False)
                
                u = ilqr(x, x_ref)
                pwm_der = torque_to_pwm(u[0])
                pwm_izq = torque_to_pwm(u[1])
                pwm_der = np.clip(pwm_der, -100, 100)
                pwm_izq = np.clip(pwm_izq, -100, 100)
                
                if trayectoria_activa == 'S':

                # Aplicar PWM mínimo
                    if 0 < abs(pwm_der) < PWM_MIN:
                        pwm_der = 40 * np.sign(pwm_der)
                    if 0 < abs(pwm_izq) < PWM_MIN:
                        pwm_izq = 40 * np.sign(pwm_izq)

                elif trayectoria_activa == 'R':
                    x_ref = generar_trayectoria_circular(derecha=True)

                                    # Aplicar PWM mínimo
                    if 0 < abs(pwm_der) < PWM_MIN:
                        pwm_der = 0 * np.sign(pwm_der)
                    if 0 < abs(pwm_izq) < PWM_MIN:
                        pwm_izq = 20 * np.sign(pwm_izq)
                    
                elif trayectoria_activa == 'L':
                            # Aplicar PWM mínimo
                    if 0 < abs(pwm_der) < PWM_MIN:
                        pwm_der = 20 * np.sign(pwm_der)
                    if 0 < abs(pwm_izq) < PWM_MIN:
                        pwm_izq = 0 * np.sign(pwm_izq)

                
                # Controlar motores
                set_motor_pwm_dir_sysfs(PWM_DER, IN1, IN2, pwm_der, period_ns)
                set_motor_pwm_dir_sysfs(PWM_IZQ, IN4, IN3, pwm_izq, period_ns)
            
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nPrograma terminado por el usuario")
    finally:
        # Limpieza final
        set_duty(PWM_DER, 0, period_ns)
        set_duty(PWM_IZQ, 0, period_ns)
        stop_pwm(PWM_DER)
        stop_pwm(PWM_IZQ)
        GPIO.cleanup()

if __name__ == '__main__':
    main()