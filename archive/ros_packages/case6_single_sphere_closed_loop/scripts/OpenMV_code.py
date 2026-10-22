import sensor, image, time, json, pyb, math
led = pyb.LED(1)
led_green = pyb.LED(2)
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
clock = time.clock()
green_threshold = (28, 72, -78, -18, -3, 43)
CENTER_X_PIXEL = 176
CENTER_Y_PIXEL = 116
SCALE_X_TO_REAL_Y = 0.068 / 91.0
SCALE_Y_TO_REAL_X = 0.068 / 92.5
def pixel_to_real(pixel_x, pixel_y):
	delta_x_pixel = pixel_x - CENTER_X_PIXEL
	delta_y_pixel = pixel_y - CENTER_Y_PIXEL
	real_x = delta_y_pixel * SCALE_Y_TO_REAL_X# main.py — FINAL WORKING VERSION (H7)
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

# === THRESHOLDS ===
thresholds = [
    (27, 74, 17, 63, -17, 48),   # red
    (28, 72, -78, -18, -3, 43),  # green
    (16, 80, -27, 0, -84, -11) # blue
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
    payload = {"red": None, "green": None, "blue": None, "fps": round(clock.fps(), 2), "t": counter}

    for b in blobs:
        if b.roundness() > 0.5:
            code = b.code()
            pos = {"x":  b.cx(), "y": b.cy(), "r": (b.w() + b.h()) // 4}
            if code == 1: payload["red"] = pos
            if code == 2: payload["green"] = pos
            if code == 4: payload["blue"] = pos

    send(payload)
    counter += 1
    time.sleep_ms(2)

	real_y = delta_x_pixel * SCALE_X_TO_REAL_Y
	return (real_x, real_y)
def rotation_from_y_axis(rotation_from_x):
	rotation_y = math.pi / 2.0 - rotation_from_x
	rotation_y = rotation_y % (2.0 * math.pi)
	rotation_deg = rotation_y * 180.0 / math.pi
	if rotation_deg > 180.0:
		rotation_deg = rotation_deg - 180.0
	return rotation_deg
send({"status": "GREEN LINE READY", "method": "USB_VCP"})
counter = 0
while True:
	clock.tick()
	img = sensor.snapshot()
	blobs = img.find_blobs([green_threshold], pixels_threshold=20, area_threshold=20, merge=True)
	best_blob = None
	max_len = 0
	for b in blobs:
		length = max(b.w(), b.h())
		if 10 < length < 60:
			if b.roundness() < 0.65:
				if length > max_len:
					max_len = length
					best_blob = b
	if best_blob:
		led.on()
		pixel_x = best_blob.cx()
		pixel_y = best_blob.cy()
		real_x, real_y = pixel_to_real(pixel_x, pixel_y)
		rotation_deg = rotation_from_y_axis(best_blob.rotation())
		pos = {
			"x": round(real_x, 6),
			"y": round(real_y, 6),
			"rotation": round(rotation_deg, 2)
		}
		payload = {
			"green": pos,
			"fps": round(clock.fps(), 2),
			"t": counter
		}
		send(payload)
	else:
		led.off()
	counter += 1
	time.sleep_ms(2)