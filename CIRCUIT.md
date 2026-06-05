# Raspberry Pi Pico W & SSD1306 OLED Circuit Diagram

This document describes the circuit design and pin mapping for the Pico W TinyLlama Chatbot project.

![Circuit Diagram](circuit_diagram.jpg)

---

## 🔌 Pin Connections Table

The SSD1306 OLED display communicates with the Raspberry Pi Pico W via the I2C0 bus interface. Connect the pins according to this table:

| SSD1306 OLED Pin | Pico W Pin Name | Pico W Physical Pin # | I2C Role / Function |
| :--- | :--- | :--- | :--- |
| **GND** | **GND** | **Pin 38** (or any GND pin) | Ground |
| **VCC** | **3V3(OUT)** | **Pin 36** | 3.3V Power Output |
| **SDA** | **GP0** | **Pin 1** | I2C0 SDA (Serial Data) |
| **SCL / SCK** | **GP1** | **Pin 2** | I2C0 SCL (Serial Clock) |

---

## 🎨 Circuit Schematic (Mermaid Diagram)

```mermaid
graph LR
    subgraph Raspberry Pi Pico W
        GP0[GPIO 0 / Pin 1]
        GP1[GPIO 1 / Pin 2]
        V3V3[3V3 OUT / Pin 36]
        GND1[GND / Pin 38]
    end

    subgraph SSD1306 OLED (128x32)
        SDA[SDA]
        SCL[SCL / SCK]
        VCC[VCC]
        GND2[GND]
    end

    GP0  <-- "I2C0 SDA (Data)" --> SDA
    GP1  <-- "I2C0 SCL (Clock)" --> SCL
    V3V3 --- "3.3V Power" ---> VCC
    GND1 --- "Ground" ---> GND2
```

---

## 💡 Engineering Notes & Design Choices
1. **Pull-up Resistors**: Most breakout boards for the SSD1306 OLED (e.g., Adafruit, SparkFun, or generic eBay/Amazon modules) already have onboard **10kΩ pull-up resistors** connected to the I2C SCL and SDA lines. If you are using a raw display panel without a breakout board, you should add two external **4.7kΩ pull-up resistors** connected from SCL and SDA to 3V3.
2. **I2C Bus Selection**: The RP2040 chip supports multiple I2C pin mappings. We configure the hardware driver in `SLM.cpp` to use **`i2c0`** on **GPIO 0** and **GPIO 1** at a baudrate of **`400 kHz`** (Fast Mode) for smooth rendering updates.
