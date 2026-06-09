# main.py - Case 8 laser spot tracker
import sensor, image, time, json, pyb


led = pyb.LED(1)
time.sleep_ms(2000)
usb = pyb.USB_VCP()


def send(data):
    try:
        usb.write((json.dumps(data) + "\n").encode())
    except:
        pass


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
sensor.set_auto_exposure(False, exposure_us=2500)
clock = time.clock()

# Tune in OpenMV IDE for the actual laser and filter. This default prefers a
# saturated red laser spot and rejects large red objects.
LASER_THRESHOLD = (35, 100, 20, 100, -15, 65)
MIN_PIXELS = 3
MIN_AREA = 3
MAX_AREA = 220
MAX_RADIUS_PX = 16


def blob_radius(blob):
    return (blob.w() + blob.h()) // 4


def blob_ok(blob):
    return blob.area() <= MAX_AREA and blob_radius(blob) <= MAX_RADIUS_PX


def blob_score(blob):
    return blob.pixels() * 2 + blob.area() - 6 * blob_radius(blob)


def find_laser_spot(img):
    best = None
    best_score = None
    blobs = img.find_blobs([LASER_THRESHOLD], pixels_threshold=MIN_PIXELS, area_threshold=MIN_AREA, merge=True)
    for blob in blobs:
        if not blob_ok(blob):
            continue
        score = blob_score(blob)
        if best is None or score > best_score:
            best = blob
            best_score = score
    return best


def blob_position(blob):
    if blob is None:
        return None
    return {"x": blob.cx(), "y": blob.cy(), "r": blob_radius(blob), "pixels": blob.pixels()}


send({"status": "CASE8 LASER READY", "method": "USB_VCP"})

counter = 0
while True:
    led.toggle()
    clock.tick()
    img = sensor.snapshot()
    spot = find_laser_spot(img)
    send({
        "spot": blob_position(spot),
        "fps": round(clock.fps(), 2),
        "t": counter,
    })
    counter += 1
    time.sleep_ms(2)
