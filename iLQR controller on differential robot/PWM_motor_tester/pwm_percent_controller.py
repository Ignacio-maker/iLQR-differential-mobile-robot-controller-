import Jetson.GPIO as GPIO
import time

# ==== Pines ====
PWM_DERECHO = 33  # BCM 12 = BOARD 32 (PWM0)
PWM_IZQUIERDO = 32  # BCM 13 = BOARD 33 (PWM2)

IN1 = 23
IN2 = 24
IN3 = 21
IN4 = 22

# ==== Inicialización GPIO ====
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)

GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT, initial=GPIO.LOW)
GPIO.setup([PWM_DERECHO, PWM_IZQUIERDO], GPIO.OUT)

# ==== PWM ====
pwm_der = GPIO.PWM(PWM_DERECHO, 100)      # Usa 100 Hz para mayor estabilidad
pwm_izq = GPIO.PWM(PWM_IZQUIERDO, 100)

pwm_der.start(0)  # Inicia en 0%
pwm_izq.start(0)

# ==== Funciones ====
def mover_adelante():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def detener():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_der.ChangeDutyCycle(0)
    pwm_izq.ChangeDutyCycle(0)

# ==== Loop de usuario ====
try:
    while True:
        entrada = input("Ingresa porcentaje PWM (0-100) o 'q' para salir: ")
        if entrada.lower() == 'q':
            break
        try:
            pwm_val = float(entrada)
            time.sleep(5)

            if 0 <= pwm_val <= 100:
                mover_adelante()
                pwm_der.ChangeDutyCycle(pwm_val)
                pwm_izq.ChangeDutyCycle(pwm_val)
                print(f"PWM aplicado: {pwm_val}%")
            else:
                print("Valor fuera de rango (0-100)")
        except ValueError:
            print("Entrada inválida.")
    detener()

finally:
    detener()
    pwm_der.stop()
    pwm_izq.stop()
    GPIO.cleanup()
    print("GPIO limpio. Programa terminado.")
