# main.py — GREEN LINE DETECTION VERSION
import sensor, image, time, json, pyb, math

# === LED ===
led = pyb.LED(1)  # Red
led_green = pyb.LED(2) # Green (Optional indication)

# === WAIT FOR USB TO SETTLE ===
time.sleep_ms(2000)  # CRITICAL: Let USB VCP initialize

# === USE USB_VCP() ===
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
# Only keeping the Green threshold
# Format: (L Min, L Max, A Min, A Max, B Min, B Max)
green_threshold = (28, 72, -78, -18, -3, 43)

# === COORDINATE TRANSFORMATION ===
# Calibration points:
# Pixel (270, 116) → Real (0, 0.068)
# Pixel (174, 26) → Real (-0.068, 0)
# Pixel (88, 116) → Real (0, -0.068)
# Pixel (176, 211) → Real (0.068, 0)
# Center appears to be around (176, 116)
# Transformation: pixel x-axis maps to real y-axis, pixel y-axis maps to real x-axis

CENTER_X_PIXEL = 176
CENTER_Y_PIXEL = 116

# Calculate scaling factors from calibration points
# From center to right: (270-176, 116-116) = (94, 0) → (0, 0.068)
# From center to left: (88-176, 116-116) = (-88, 0) → (0, -0.068)
# Use average distance: (94 + 88) / 2 = 91 pixels → 0.068 meters
SCALE_X_TO_REAL_Y = 0.068 / 91.0  # pixel x → real y

# From center to top: (174-176, 26-116) = (-2, -90) → (-0.068, 0)
# From center to bottom: (176-176, 211-116) = (0, 95) → (0.068, 0)
# Use exact distances: 90 pixels → 0.068m and 95 pixels → 0.068m
# Average: (90 + 95) / 2 = 92.5 pixels → 0.068 meters
SCALE_Y_TO_REAL_X = 0.068 / 92.5  # pixel y → real x

def pixel_to_real(pixel_x, pixel_y):
    """
    Convert pixel coordinates to real-world coordinates.
    Args:
        pixel_x: X coordinate in pixels
        pixel_y: Y coordinate in pixels
    Returns:
        Tuple (real_x, real_y) in meters
    """
    # Calculate offset from center
    delta_x_pixel = pixel_x - CENTER_X_PIXEL
    delta_y_pixel = pixel_y - CENTER_Y_PIXEL
    
    # Transform: pixel x → real y, pixel y → real x (with sign adjustment)
    real_x = delta_y_pixel * SCALE_Y_TO_REAL_X
    real_y = delta_x_pixel * SCALE_X_TO_REAL_Y
    
    return (real_x, real_y)

def rotation_from_y_axis(rotation_from_x):
    """
    Convert rotation angle from x-axis reference to y-positive axis reference.
    In pixel coordinates, positive y-axis points downward.
    Args:
        rotation_from_x: Rotation angle in radians from horizontal (x-axis)
    Returns:
        Rotation angle in degrees (0-180) from y-positive axis (downward)
        - 0 degrees: line parallel to y-positive axis (vertical down)
        - 90 degrees: line parallel to x-positive axis (horizontal right)
        - 180 degrees: line parallel to y-negative axis (vertical up)
    """
    # Convert from x-axis reference to y-positive axis reference: subtract π/2
    # This gives angle from positive y-axis (downward in pixel coords)
    rotation_y = math.pi / 2.0 - rotation_from_x
    
    # Normalize to [0, 2π) range
    rotation_y = rotation_y % (2.0 * math.pi)
    
    # Convert to degrees
    rotation_deg = rotation_y * 180.0 / math.pi
    
    # Normalize to [0, 180] range (since line orientation is modulo 180)
    if rotation_deg > 180.0:
        rotation_deg = rotation_deg - 180.0
    
    return rotation_deg

# === STARTUP ===
send({"status": "GREEN LINE READY", "method": "USB_VCP"})

# === MAIN LOOP ===
counter = 0
while True:
    clock.tick()
    img = sensor.snapshot()

    # 1. Find Blobs
    # We lower the pixels_threshold and area_threshold because a "thin line"
    # has much less total area than a solid sphere.
    blobs = img.find_blobs([green_threshold], pixels_threshold=20, area_threshold=20, merge=True)

    best_blob = None
    max_len = 0

    for b in blobs:
        # 2. Analyze Shape
        # A line or thin rectangle is elongated.
        # We calculate the "length" as the maximum of width or height.
        length = max(b.w(), b.h())

        # 3. Filter by Size (Target is around 30px)
        # We accept a range (e.g., 15 to 60) to account for distance/noise.
        if 10 < length < 60:

            # Optional: Check for elongation (lines have low roundness)
            # If roundness is high (>0.6), it's a circle, not a line.
            if b.roundness() < 0.65:

                # If multiple lines are found, pick the largest one
                if length > max_len:
                    max_len = length
                    best_blob = b

    # 4. Process the best match - only send data if detected
    if best_blob:
        led.on() # LED on if detected

        # Calculate pixel position
        pixel_x = best_blob.cx()
        pixel_y = best_blob.cy()
        
        # Convert to real-world coordinates
        real_x, real_y = pixel_to_real(pixel_x, pixel_y)
        
        # Calculate rotation angle from y-positive axis (downward in pixel coords, 0-180 degrees)
        rotation_deg = rotation_from_y_axis(best_blob.rotation())
        
        # Calculate position (now with real coordinates)
        pos = {
            "x": round(real_x, 6),  # Real-world X coordinate in meters
            "y": round(real_y, 6),  # Real-world Y coordinate in meters
            # "pixel_x": pixel_x,     # Pixel X for reference
            # "pixel_y": pixel_y,     # Pixel Y for reference
            # "w": best_blob.w(),
            # "h": best_blob.h(),
            "rotation": round(rotation_deg, 2)  # Angle from y-positive axis (downward) in degrees (0-180)
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
