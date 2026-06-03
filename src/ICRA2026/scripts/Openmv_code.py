# main.py - OpenMV H7 two-microsphere tracker
import sensor, image, time, json, pyb

# === LED ===
led = pyb.LED(1)  # Red

# === WAIT FOR USB TO SETTLE ===
time.sleep_ms(2000)  # CRITICAL: Let USB VCP initialize

# === USE USB_VCP() - WORKS EVERY TIME ===
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

# === DETECTION SETTINGS ===
# OpenMV LAB thresholds. These must usually be tuned in OpenMV IDE for the
# actual lighting, camera exposure, and marker material.
RED_THRESHOLD = (20, 90, 20, 80, 0, 70)
GREEN_THRESHOLD = (28, 72, -78, -18, -3, 43)
THRESHOLDS = [
    RED_THRESHOLD,    # code 1
    GREEN_THRESHOLD,  # code 2
]

MIN_ROUNDNESS = 0.5
MIN_PIXELS = 100
MIN_AREA = 100

# Physical radius is 4 mm. Radius filtering needs calibration because OpenMV
# sees pixels, not mm. Leave PIXELS_PER_MM as None to disable radius filtering.
SPHERE_RADIUS_MM = 4.0
PIXELS_PER_MM = None
RADIUS_TOLERANCE = 0.35


def radius_ok(radius_px):
    if PIXELS_PER_MM is None:
        return True
    expected = SPHERE_RADIUS_MM * PIXELS_PER_MM
    return expected * (1.0 - RADIUS_TOLERANCE) <= radius_px <= expected * (1.0 + RADIUS_TOLERANCE)


def score_blob(blob):
    # Prefer large, round blobs when there are several objects of the same color.
    return blob.pixels() * blob.roundness()


def update_best(payload, color, blob):
    radius_px = (blob.w() + blob.h()) // 4
    if blob.roundness() < MIN_ROUNDNESS or not radius_ok(radius_px):
        return

    candidate = {
        "x": blob.cx(),
        "y": blob.cy(),
        "r": radius_px,
        "score": score_blob(blob),
    }
    current = payload[color]
    if current is None or candidate["score"] > current["score"]:
        payload[color] = candidate

# === STARTUP ===
send({"status": "H7 READY", "method": "USB_VCP"})

# === MAIN LOOP ===
counter = 0
while True:
    led.toggle()
    clock.tick()
    img = sensor.snapshot()

    blobs = img.find_blobs(THRESHOLDS, pixels_threshold=MIN_PIXELS, area_threshold=MIN_AREA, merge=True)
    payload = {"red": None, "green": None, "fps": round(clock.fps(), 2), "t": counter}

    for b in blobs:
        code = b.code()
        if code == 1:
            update_best(payload, "red", b)
        elif code == 2:
            update_best(payload, "green", b)

    for color in ("red", "green"):
        if payload[color] is not None:
            del payload[color]["score"]

    send(payload)
    counter += 1
    time.sleep_ms(2)
