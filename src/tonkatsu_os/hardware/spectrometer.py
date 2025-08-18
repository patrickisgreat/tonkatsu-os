"""
Hardware interface for B&W Tek Raman spectrometers.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import serial

logger = logging.getLogger(__name__)


@dataclass
class SpectrometerConfig:
    """Configuration for B&W Tek spectrometer."""

    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    timeout: float = 2.0
    data_points: int = 2048
    default_integration_time: int = 200  # milliseconds


class BWTekSpectrometer:
    """
    Interface for B&W Tek Raman spectrometers.

    Based on the original working protocol:
    - ASCII mode command: 'a\r\n'
    - Integration time: 'I{time}\r\n' (time in ms)
    - Start acquisition: 'S\r\n'
    """

    def __init__(self, config: SpectrometerConfig = None):
        self.config = config or SpectrometerConfig()
        self._serial_connection: Optional[serial.Serial] = None
        self._is_connected = False
        self._last_communication = None
        self._temperature = None

    def connect(self, port: str = None) -> bool:
        """Connect to the spectrometer."""
        if port:
            self.config.port = port

        try:
            self._serial_connection = serial.Serial(
                port=self.config.port, baudrate=self.config.baudrate, timeout=self.config.timeout
            )

            # Test communication
            self._serial_connection.write(b"a\r\n")  # ASCII mode
            time.sleep(0.1)

            self._is_connected = True
            self._last_communication = datetime.now()

            logger.info(f"Connected to B&W Tek spectrometer on {self.config.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to spectrometer: {e}")
            self._is_connected = False
            if self._serial_connection:
                self._serial_connection.close()
                self._serial_connection = None
            return False

    def disconnect(self) -> bool:
        """Disconnect from the spectrometer."""
        try:
            if self._serial_connection and self._serial_connection.is_open:
                self._serial_connection.close()

            self._serial_connection = None
            self._is_connected = False
            self._last_communication = None

            logger.info("Disconnected from spectrometer")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to spectrometer."""
        if not self._is_connected or not self._serial_connection:
            return False

        try:
            # Check if serial port is still open
            return self._serial_connection.is_open
        except:
            return False

    def acquire_spectrum(self, integration_time: int = None) -> Optional[np.ndarray]:
        """
        Acquire spectrum from the spectrometer.

        Args:
            integration_time: Integration time in milliseconds

        Returns:
            Numpy array of spectrum data or None if failed
        """
        if not self.is_connected():
            logger.error("Spectrometer not connected")
            return None

        if integration_time is None:
            integration_time = self.config.default_integration_time

        try:
            # Send commands following B&W Tek protocol
            self._serial_connection.write(b"a\r\n")  # ASCII mode
            time.sleep(0.1)

            # Set integration time
            integration_cmd = f"I{integration_time}\r\n".encode()
            self._serial_connection.write(integration_cmd)
            time.sleep(0.1)

            # Start acquisition
            self._serial_connection.write(b"S\r\n")

            # Read spectrum data
            raw_data = self._serial_connection.read_until(expected=b"\r\n")

            # Parse spectrum data
            spectrum_str = raw_data.decode(errors="ignore").strip()
            spectrum_values = spectrum_str.split()

            if len(spectrum_values) < self.config.data_points:
                logger.warning(
                    f"Received {len(spectrum_values)} data points, expected {self.config.data_points}"
                )

            # Convert to numpy array, pad if necessary
            spectrum = np.array([int(x) for x in spectrum_values[: self.config.data_points]])

            if len(spectrum) < self.config.data_points:
                # Pad with zeros if we got fewer points than expected
                padded = np.zeros(self.config.data_points)
                padded[: len(spectrum)] = spectrum
                spectrum = padded

            self._last_communication = datetime.now()

            logger.info(
                f"Acquired spectrum with {len(spectrum)} points, integration time {integration_time}ms"
            )
            return spectrum

        except Exception as e:
            logger.error(f"Error acquiring spectrum: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get spectrometer status."""
        return {
            "connected": self.is_connected(),
            "port": self.config.port if self.is_connected() else None,
            "baudrate": self.config.baudrate,
            "last_communication": self._last_communication,
            "temperature": self._temperature,
            "data_points": self.config.data_points,
        }

    def laser_on(self) -> bool:
        """Turn on laser (if supported by hardware)."""
        # B&W Tek protocol might not have direct laser control
        # This would need to be implemented based on specific hardware
        logger.info("Laser control not implemented for this hardware")
        return True

    def laser_off(self) -> bool:
        """Turn off laser (if supported by hardware)."""
        # B&W Tek protocol might not have direct laser control
        logger.info("Laser control not implemented for this hardware")
        return True


class HardwareManager:
    """Manager for multiple hardware interfaces."""

    def __init__(self):
        self.spectrometer: Optional[BWTekSpectrometer] = None
        self._available_ports = []

    def scan_ports(self) -> list:
        """Scan for available serial ports."""
        import serial.tools.list_ports

        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append(
                {"device": port.device, "description": port.description, "hwid": port.hwid}
            )

        self._available_ports = ports
        return ports

    def connect_spectrometer(self, port: str = "/dev/ttyUSB0") -> bool:
        """Connect to spectrometer."""
        if self.spectrometer:
            self.spectrometer.disconnect()

        config = SpectrometerConfig(port=port)
        self.spectrometer = BWTekSpectrometer(config)

        return self.spectrometer.connect()

    def disconnect_spectrometer(self) -> bool:
        """Disconnect spectrometer."""
        if self.spectrometer:
            return self.spectrometer.disconnect()
        return True

    def get_spectrometer_status(self) -> Dict[str, Any]:
        """Get spectrometer status."""
        if self.spectrometer:
            return self.spectrometer.get_status()

        return {"connected": False, "port": None, "last_communication": None, "temperature": None}

    def acquire_spectrum(self, integration_time: int = None) -> Optional[np.ndarray]:
        """Acquire spectrum from connected spectrometer."""
        if not self.spectrometer:
            logger.error("No spectrometer connected")
            return None

        return self.spectrometer.acquire_spectrum(integration_time)
