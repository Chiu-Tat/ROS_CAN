# main.py — FINAL WORKING VERSION (H7)
import sensor, image, time, json, pyb

# === LED ===
led = pyb.LED(1)  # Red

# === WAIT FOR USB TO SETTLE ===
time.sleep_ms(2000)  # CRITICAL: Let USB VCP initialize

# === USE USB_VCP() — WORKS EVERY TIME ===
usb = pyb.USB_VCP()
def send(data):
    try:
        usb.write((json.dumps(data) + "\n").encode())
    except:
        pass  # Ignore USB errors

# === SENSOR ===
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
clock = time.clock()

# === THRESHOLDS (green and blue only) ===
thresholds = [
    (28, 72, -78, -18, -3, 43),   # green
    (16, 80, -27, 0, -84, -11)    # blue
]

# === STARTUP ===
send({"status": "H7 READY", "method": "USB_VCP"})

# === MAIN LOOP ===
counter = 0
while True:
    led.toggle()
    clock.tick()
    img = sensor.snapshot()

    blobs = img.find_blobs(thresholds, pixels_threshold=100, area_threshold=100, merge=True)
    payload = {"green": None, "blue": None, "fps": round(clock.fps(), 2), "t": counter}

    for b in blobs:
        if b.roundness() > 0.5:
            code = b.code()
            pos = {"x": b.cx(), "y": b.cy(), "r": (b.w() + b.h()) // 4}
            if code == 1: payload["green"] = pos
            if code == 2: payload["blue"] = pos

    send(payload)
    counter += 1
    time.sleep_ms(2)