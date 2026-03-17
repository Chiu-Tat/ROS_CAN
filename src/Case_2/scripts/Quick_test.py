import serial
import json
import glob
import time

# Find port
ports = glob.glob('/dev/ttyACM*')
if not ports:
    print("No OpenMV found!")
    exit()

port = ports[0]
print(f"Connecting to {port}...")

# Wait for boot
time.sleep(3)

ser = serial.Serial(port, 115200, timeout=1)
print("Listening... (Ctrl+C to stop)")

while True:
    line = ser.readline()
    if line:
        try:
            text = line.decode('utf-8', errors='ignore').strip()
            if text:
                data = json.loads(text)
                print("→", data)
                if data.get("red"):
                    r = data["red"]
                    print(f"   RED SPHERE: ({r['x']}, {r['y']}), r={r['r']}")
        except:
            print("Bad:", line)