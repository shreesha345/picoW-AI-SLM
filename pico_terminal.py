import sys
import time
import serial
import threading
import serial.tools.list_ports

def find_ports():
    ports = list(serial.tools.list_ports.comports())
    return ports

def read_thread(ser, stop_event):
    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                # Decode bytes to string, handling standard console characters
                sys.stdout.write(data.decode('utf-8', errors='ignore'))
                sys.stdout.flush()
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"\n[Connection lost: {e}]")
            stop_event.set()
            break

def main():
    print("=========================================")
    print("       Pico W Serial Terminal Client     ")
    print("=========================================")
    
    ports = find_ports()
    if not ports:
        print("No serial ports found! Please check that your Pico W is plugged in.")
        input("Press Enter to exit...")
        return
        
    print("Available ports:")
    for idx, port in enumerate(ports):
        print(f"[{idx}] {port.device} - {port.description}")
        
    if len(ports) == 1:
        choice = 0
        print(f"\nAutomatically selecting only available port: {ports[0].device}")
    else:
        try:
            choice_str = input(f"\nSelect port [0-{len(ports)-1}]: ")
            choice = int(choice_str)
        except ValueError:
            print("Invalid input, defaulting to 0.")
            choice = 0
            
    port_name = ports[choice].device
    baudrate = 115200
    
    print(f"Connecting to {port_name} at {baudrate} baud...")
    try:
        ser = serial.Serial(port_name, baudrate, timeout=1)
    except Exception as e:
        print(f"Failed to connect to {port_name}: {e}")
        input("Press Enter to exit...")
        return
        
    print("Connected! Press Ctrl+C to exit.")
    print("-----------------------------------------")
    
    stop_event = threading.Event()
    t = threading.Thread(target=read_thread, args=(ser, stop_event), daemon=True)
    t.start()
    
    # Use msvcrt for character-by-character interactive console input (Windows native)
    import msvcrt
    try:
        while not stop_event.is_set():
            if msvcrt.kbhit():
                char = msvcrt.getch()
                
                # Check for Ctrl+C to exit terminal script locally
                if char == b'\x03':
                    print("\nExiting terminal client...")
                    break
                    
                ser.write(char)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nExiting terminal client...")
    finally:
        stop_event.set()
        ser.close()
        t.join(timeout=1)

if __name__ == "__main__":
    main()
