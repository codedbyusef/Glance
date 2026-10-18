# Glance glove (CircuitPython, ESP32-S3): runs once at hard reset, before code.py.
#
# Swaps the REPL console for a raw "data" USB serial port. Motor frames need a raw port because
# on the console a 0x03 byte acts as Ctrl+C and would stop code.py.
#
# The S3 has too few USB endpoints for console + data + CIRCUITPY drive (it boots into safe mode
# with "USB devices need more endpoints than are available"), so the console is turned off and
# the drive is kept. The data port is then the only serial port, usually /dev/ttyACM0.
# To get the REPL back: delete or rename boot.py on CIRCUITPY and press RESET.
import usb_cdc
import usb_hid
import usb_midi

usb_hid.disable()
usb_midi.disable()
usb_cdc.enable(console=False, data=True)
