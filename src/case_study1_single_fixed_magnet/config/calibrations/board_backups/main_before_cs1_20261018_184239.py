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
clock = time.clock()
RED_THRESHOLD = (20, 90, 20, 80, -20, 70)
GREEN_THRESHOLD = (28, 72, -78, -14, -9, 43)
MIN_ROUNDNESS = 0.5
MIN_PIXELS = 50
MIN_AREA = 50
EXPECTED_RADIUS_PX = 7
MIN_RADIUS_PX = 4
MAX_RADIUS_PX = 12
def blob_radius(blob):
	return (blob.w() + blob.h()) // 4
def blob_ok(blob):
	radius = blob_radius(blob)
	return (
		blob.roundness() >= MIN_ROUNDNESS and
		MIN_RADIUS_PX <= radius <= MAX_RADIUS_PX
	)
def blob_score(blob):
	radius = blob_radius(blob)
	radius_error = abs(radius - EXPECTED_RADIUS_PX)
	return blob.pixels() * blob.roundness() - 20 * radius_error
def find_best_blob(img, threshold):
	best = None
	best_score = None
	blobs = img.find_blobs([threshold], pixels_threshold=MIN_PIXELS, area_threshold=MIN_AREA, merge=True)
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
	return {"x": blob.cx(), "y": blob.cy(), "r": blob_radius(blob)}
send({"status": "H7 READY", "method": "USB_VCP"})
counter = 0
while True:
	led.toggle()
	clock.tick()
	img = sensor.snapshot()
	red_blob = find_best_blob(img, RED_THRESHOLD)
	green_blob = find_best_blob(img, GREEN_THRESHOLD)
	payload = {
		"red": blob_position(red_blob),
		"green": blob_position(green_blob),
		"fps": round(clock.fps(), 2),
		"t": counter,
	}
	send(payload)
	counter += 1
	time.sleep_ms(2)