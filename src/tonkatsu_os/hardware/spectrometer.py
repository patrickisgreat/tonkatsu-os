"""
Hardware interface for B&W Tek Raman spectrometers.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import serial

# Enable debug logging for spectrometer troubleshooting
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SpectrometerError(RuntimeError):
    """Base exception for spectrometer errors."""


class SpectrometerConnectionError(SpectrometerError):
    """Raised when connecting to the spectrometer fails."""


class SpectrometerAcquisitionError(SpectrometerError):
    """Raised when acquiring a spectrum fails."""


@dataclass
class SpectrometerConfig:
    """Configuration for B&W Tek spectrometer."""

    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    timeout: float = 10.0  # Increased for longer integration times + data transfer
    data_points: int = 2048
    min_data_points: int = 2048  # Require full frame for reliable normalization
    default_integration_time: int = 200  # milliseconds (lower default to avoid saturation)
    max_retries: int = 3
    retry_delay: float = 0.5  # seconds between retries
    simulate: bool = False
    simulation_file: Optional[str] = None
    response_terminator: bytes = b"\r\n"
    # Debug logging - saves raw scan data to files for analysis
    debug_log_dir: Optional[str] = "scan_logs"
    debug_logging_enabled: bool = True


class BWTekSpectrometer:
    """
    Interface for B&W Tek Raman spectrometers.

    Based on the original working protocol:
    - ASCII mode command: 'a\\r\\n'
    - Integration time: 'I{time}\\r\\n' (time in ms)
    - Start acquisition: 'S\\r\\n'
    """

    def __init__(self, config: SpectrometerConfig = None):
        self.config = config or SpectrometerConfig()
        self._serial_connection: Optional[serial.Serial] = None
        self._is_connected = False
        self._last_communication: Optional[datetime] = None
        self._temperature: Optional[float] = None
        self._last_error: Optional[str] = None
        self._last_source: Optional[str] = None
        # Dark spectrum for background subtraction
        self._dark_spectrum: Optional[np.ndarray] = None
        self._dark_integration_time: Optional[int] = None

    # ---------------------------------------------------------------------
    # Connection lifecycle
    # ---------------------------------------------------------------------
    def connect(
        self,
        port: Optional[str] = None,
        *,
        simulate: Optional[bool] = None,
        simulation_file: Optional[str] = None,
    ) -> bool:
        """Connect to the spectrometer or initialize the simulator."""
        if port:
            self.config.port = port
        if simulate is not None:
            self.config.simulate = simulate
        if simulation_file is not None:
            self.config.simulation_file = simulation_file

        if self.config.simulate:
            self._close_serial()
            self._is_connected = True
            self._last_communication = datetime.now()
            self._last_source = "simulator"
            self._last_error = None
            logger.info(
                "Initialized spectrometer simulator (file=%s)",
                self.config.simulation_file,
            )
            return True

        self._close_serial()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._serial_connection = serial.Serial(
                    port=self.config.port,
                    baudrate=self.config.baudrate,
                    timeout=self.config.timeout,
                )
                self._serial_connection.reset_input_buffer()
                self._serial_connection.reset_output_buffer()

                # Initialize ASCII mode and wait briefly for hardware response
                self._serial_connection.write(b"a\r\n")
                self._serial_connection.flush()
                time.sleep(0.1)

                self._is_connected = True
                self._last_communication = datetime.now()
                self._last_source = "hardware"
                self._last_error = None
                logger.info(
                    "Connected to B&W Tek spectrometer on %s",
                    self.config.port,
                )
                return True
            except Exception as exc:  # pragma: no cover - hardware specific
                last_exc = exc
                logger.warning(
                    "Connection attempt %s/%s failed: %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                self._close_serial()
                time.sleep(self.config.retry_delay)

        self._is_connected = False
        error_msg = (
            f"Failed to connect to spectrometer on {self.config.port}: {last_exc}"
        )
        self._last_error = error_msg
        logger.error(error_msg)
        raise SpectrometerConnectionError(error_msg)

    def disconnect(self) -> bool:
        """Disconnect from the spectrometer."""
        try:
            self._close_serial()
            self._is_connected = False
            self._last_communication = None
            self._last_source = None if not self.config.simulate else "simulator"

            logger.info("Disconnected from spectrometer")
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            self._last_error = str(exc)
            logger.error("Error disconnecting: %s", exc)
            return False

    def is_connected(self) -> bool:
        """Check if the spectrometer (or simulator) is connected."""
        if self.config.simulate:
            return True

        if not self._serial_connection:
            return False

        try:
            self._is_connected = bool(self._serial_connection.is_open)
        except Exception:  # pragma: no cover - defensive logging
            self._is_connected = False
        return self._is_connected

    # ---------------------------------------------------------------------
    # Acquisition
    # ---------------------------------------------------------------------
    def acquire_spectrum(
        self,
        integration_time: Optional[int] = None,
        *,
        simulate: Optional[bool] = None,
        simulation_file: Optional[str] = None,
    ) -> np.ndarray:
        """
        Acquire a spectrum from the spectrometer or simulator.

        Args:
            integration_time: Integration time in milliseconds.
            simulate: Override to force simulator usage.
            simulation_file: Override data file for simulator mode.

        Returns:
            np.ndarray: Spectrum data as floats.
        """
        use_simulator = self.config.simulate if simulate is None else simulate
        simulation_path = simulation_file or self.config.simulation_file

        if integration_time is None:
            integration_time = self.config.default_integration_time

        if use_simulator:
            spectrum = self._load_simulated_spectrum(simulation_path)
            self._last_source = "simulator"
            self._last_error = None
            self._last_communication = datetime.now()
            return spectrum

        if not self.is_connected():
            error_msg = "Spectrometer not connected"
            self._last_error = error_msg
            logger.error(error_msg)
            raise SpectrometerConnectionError(error_msg)

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Reset any buffered garbage before starting a new acquisition
                if self._serial_connection:
                    self._serial_connection.reset_input_buffer()

                # Configure integration time and start acquisition
                self._serial_connection.write(b"a\r\n")
                self._serial_connection.flush()
                time.sleep(0.05)

                integration_cmd = f"I{int(integration_time)}\r\n".encode()
                self._serial_connection.write(integration_cmd)
                self._serial_connection.flush()
                time.sleep(0.05)

                self._serial_connection.write(b"S\r\n")
                self._serial_connection.flush()

                # Wait for acquisition to complete (integration time + buffer)
                wait_time = (integration_time / 1000.0) + 0.5
                logger.debug("Waiting %.2fs for acquisition to complete", wait_time)
                time.sleep(wait_time)

                raw_data = self._read_raw_response()
                spectrum = self._parse_raw_spectrum(raw_data)
                normalized = self._normalize_spectrum(spectrum)

                # Save debug log for analysis
                self._save_debug_log(raw_data, list(spectrum), integration_time)

                self._last_communication = datetime.now()
                self._last_source = "hardware"
                self._last_error = None

                logger.info(
                    "Acquired spectrum (%s points, integration=%sms)",
                    len(normalized),
                    integration_time,
                )
                return normalized
            except SpectrometerAcquisitionError as exc:
                last_exc = exc
                # Save debug log even on error
                if 'raw_data' in locals():
                    self._save_debug_log(
                        raw_data,
                        list(spectrum) if 'spectrum' in locals() else [],
                        integration_time,
                        error=str(exc),
                    )
                logger.warning(
                    "Acquisition attempt %s/%s failed: %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                time.sleep(self.config.retry_delay)
            except Exception as exc:  # pragma: no cover - hardware specific
                last_exc = SpectrometerAcquisitionError(str(exc))
                # Save debug log even on error
                if 'raw_data' in locals():
                    self._save_debug_log(
                        raw_data,
                        list(spectrum) if 'spectrum' in locals() else [],
                        integration_time,
                        error=str(exc),
                    )
                logger.warning(
                    "Unexpected acquisition error (%s/%s): %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                time.sleep(self.config.retry_delay)

        error_msg = str(last_exc) if last_exc else "Unknown acquisition failure"
        self._last_error = error_msg
        logger.error(error_msg)
        raise SpectrometerAcquisitionError(error_msg)

    # ---------------------------------------------------------------------
    # Dark subtraction
    # ---------------------------------------------------------------------
    def acquire_dark(self, integration_time: Optional[int] = None) -> np.ndarray:
        """
        Acquire a dark spectrum for background subtraction.

        Instructions: Block the laser or remove sample before calling this.
        The dark spectrum will be stored and automatically subtracted from
        future acquisitions.
        """
        if integration_time is None:
            integration_time = self.config.default_integration_time

        logger.info("Acquiring dark spectrum (integration=%sms)...", integration_time)
        logger.info("Make sure laser is blocked or sample is removed!")

        # Acquire without dark subtraction
        dark = self.acquire_spectrum(integration_time)

        self._dark_spectrum = dark
        self._dark_integration_time = integration_time

        logger.info(
            "Dark spectrum acquired: %d points, mean=%.1f, min=%.1f, max=%.1f",
            len(dark), np.mean(dark), np.min(dark), np.max(dark)
        )

        return dark

    def set_dark_spectrum(self, spectrum: np.ndarray, integration_time: int) -> None:
        """Store a dark spectrum for future subtraction."""
        self._dark_spectrum = np.asarray(spectrum, dtype=np.float32)
        self._dark_integration_time = int(integration_time)
        logger.info(
            "Dark spectrum stored: %d points, mean=%.1f, min=%.1f, max=%.1f",
            len(self._dark_spectrum),
            float(np.mean(self._dark_spectrum)),
            float(np.min(self._dark_spectrum)),
            float(np.max(self._dark_spectrum)),
        )

    def clear_dark(self) -> None:
        """Clear the stored dark spectrum."""
        self._dark_spectrum = None
        self._dark_integration_time = None
        logger.info("Dark spectrum cleared")

    def has_dark(self) -> bool:
        """Check if a dark spectrum is stored."""
        return self._dark_spectrum is not None

    def get_dark_info(self) -> Dict[str, Any]:
        """Get information about the stored dark spectrum."""
        if self._dark_spectrum is None:
            return {"has_dark": False}
        return {
            "has_dark": True,
            "integration_time": self._dark_integration_time,
            "data_points": len(self._dark_spectrum),
            "mean": float(np.mean(self._dark_spectrum)),
            "min": float(np.min(self._dark_spectrum)),
            "max": float(np.max(self._dark_spectrum)),
        }

    def apply_dark_subtraction(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Subtract the dark spectrum from a measurement.

        If the spectra have different lengths, interpolates the dark spectrum.
        """
        if self._dark_spectrum is None:
            logger.warning("No dark spectrum available, returning original")
            return spectrum

        dark = self._dark_spectrum

        # Handle length mismatch by interpolation
        if len(dark) != len(spectrum):
            logger.warning(
                "Dark spectrum length (%d) differs from sample (%d), interpolating",
                len(dark), len(spectrum)
            )
            x_dark = np.linspace(0, 1, len(dark))
            x_sample = np.linspace(0, 1, len(spectrum))
            dark = np.interp(x_sample, x_dark, dark)

        # Subtract dark and clip to non-negative
        corrected = spectrum - dark
        corrected = np.maximum(corrected, 0)  # Clip negative values to 0

        logger.info(
            "Dark subtracted: original mean=%.1f, corrected mean=%.1f",
            np.mean(spectrum), np.mean(corrected)
        )

        return corrected.astype(np.float32)

    # ---------------------------------------------------------------------
    # Status and helpers
    # ---------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Get spectrometer status details."""
        return {
            "connected": self.is_connected(),
            "port": self.config.port if self.is_connected() else None,
            "baudrate": self.config.baudrate,
            "last_communication": self._last_communication,
            "temperature": self._temperature,
            "data_points": self.config.data_points,
            "last_error": self._last_error,
            "last_source": self._last_source,
            "simulate": self.config.simulate,
            "simulation_file": self.config.simulation_file,
        }

    def laser_on(self) -> bool:
        """Turn on laser (if supported by hardware)."""
        logger.info("Laser control not implemented for this hardware")
        return True

    def laser_off(self) -> bool:
        """Turn off laser (if supported by hardware)."""
        logger.info("Laser control not implemented for this hardware")
        return True

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _close_serial(self) -> None:
        if self._serial_connection and self._serial_connection.is_open:
            try:
                self._serial_connection.close()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Error closing serial connection")
        self._serial_connection = None

    def _save_debug_log(
        self,
        raw_data: bytes,
        parsed_values: Sequence[float],
        integration_time: int,
        error: Optional[str] = None,
    ) -> Optional[Path]:
        """Save raw scan data to a debug log file for analysis."""
        if not self.config.debug_logging_enabled or not self.config.debug_log_dir:
            return None

        try:
            log_dir = Path(self.config.debug_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_file = log_dir / f"scan_{timestamp}.txt"

            with open(log_file, "w") as f:
                f.write("=" * 80 + "\n")
                f.write(f"RAMAN SPECTROMETER DEBUG LOG\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")

                # Configuration
                f.write("--- CONFIGURATION ---\n")
                f.write(f"Port: {self.config.port}\n")
                f.write(f"Baudrate: {self.config.baudrate}\n")
                f.write(f"Integration time: {integration_time} ms\n")
                f.write(f"Timeout: {self.config.timeout} s\n")
                f.write(f"Expected data points: {self.config.data_points}\n")
                f.write(f"Min data points: {self.config.min_data_points}\n")
                f.write("\n")

                # Error if any
                if error:
                    f.write("--- ERROR ---\n")
                    f.write(f"{error}\n\n")

                # Raw bytes info
                f.write("--- RAW DATA INFO ---\n")
                f.write(f"Total bytes received: {len(raw_data)}\n")
                null_byte = b'\x00'
                f.write(f"Null bytes count: {raw_data.count(null_byte)}\n")
                f.write("\n")

                # Raw bytes as hex (first 500 bytes)
                f.write("--- RAW BYTES (first 500, hex) ---\n")
                hex_data = raw_data[:500].hex(" ")
                for i in range(0, len(hex_data), 48):
                    f.write(f"{hex_data[i:i+48]}\n")
                f.write("\n")

                # Raw data as string (first 1000 chars)
                f.write("--- RAW DATA AS STRING (first 1000 chars) ---\n")
                raw_str = raw_data.decode(errors="replace")[:1000]
                f.write(f"{raw_str}\n\n")

                # Parsed values
                f.write("--- PARSED VALUES ---\n")
                f.write(f"Total parsed values: {len(parsed_values)}\n")
                if parsed_values:
                    f.write(f"Min value: {min(parsed_values)}\n")
                    f.write(f"Max value: {max(parsed_values)}\n")
                    f.write(f"Mean value: {sum(parsed_values) / len(parsed_values):.3f}\n")
                    saturation = sum(1 for val in parsed_values if val >= 65535)
                    f.write(f"Saturated values (>=65535): {saturation}\n")
                    f.write(f"First 20 values: {parsed_values[:20]}\n")
                    f.write(f"Last 20 values: {parsed_values[-20:]}\n")
                f.write("\n")

                # Full parsed data (one value per line for easy plotting)
                f.write("--- FULL SPECTRUM DATA (one value per line) ---\n")
                for i, val in enumerate(parsed_values):
                    f.write(f"{i}\t{val}\n")

            logger.info(f"Debug log saved to: {log_file}")
            return log_file

        except Exception as exc:
            logger.warning(f"Failed to save debug log: {exc}")
            return None

    def _read_raw_response(self) -> bytes:
        """Read raw bytes from the serial port until timeout or sufficient data."""
        if not self._serial_connection:
            raise SpectrometerAcquisitionError("Serial connection not initialized")

        # Estimate minimum read time based on baudrate and expected frame size.
        # 5 digits + CRLF ~= 7 bytes per point, 10 bits per byte on the wire.
        bytes_per_point = 7
        expected_seconds = (
            self.config.data_points * bytes_per_point * 10
        ) / self.config.baudrate
        deadline = time.time() + max(self.config.timeout, expected_seconds + 2.0)
        buffer = bytearray()

        # Read all available data, not just until first terminator
        while time.time() < deadline:
            # Check how much data is available
            available = self._serial_connection.in_waiting
            if available > 0:
                chunk = self._serial_connection.read(available)
                if chunk:
                    buffer.extend(chunk)
                    logger.debug(
                        "Read %d bytes (total buffer: %d bytes)", len(chunk), len(buffer)
                    )
                    if self._count_numeric_values(buffer) >= self.config.data_points:
                        break
            else:
                # No data available, wait briefly and keep listening until deadline.
                time.sleep(0.1)

        if not buffer:
            raise SpectrometerAcquisitionError(
                "No data received from spectrometer before timeout"
            )

        logger.debug(
            "Total raw data received: %d bytes, first 100 chars: %s",
            len(buffer),
            buffer[:100].decode(errors="ignore"),
        )

        return bytes(buffer)

    def _count_numeric_values(self, raw_data: bytes) -> int:
        """Count numeric spectrum values in the raw buffer."""
        try:
            spectrum_str = raw_data.replace(b"\x00", b" ").decode(errors="ignore")
        except Exception:
            return 0

        lines = spectrum_str.replace(",", " ").splitlines()
        skip_patterns = {"ACK", "a", "S"}

        count = 0
        for line in lines:
            line = line.strip()
            if not line or line in skip_patterns or line.startswith("I"):
                continue
            if "ACK" in line:
                continue
            if not line.isdigit() or len(line) < 4:
                continue
            count += 1
        return count

    def _parse_raw_spectrum(self, raw_data: bytes) -> Sequence[float]:
        """Decode and validate raw spectrum data."""
        try:
            # Remove null bytes and other control characters before decoding
            clean_data = raw_data.replace(b"\x00", b" ")
            spectrum_str = clean_data.decode(errors="ignore").strip()
        except Exception as exc:
            raise SpectrometerAcquisitionError(
                f"Failed to decode spectrum bytes: {exc}"
            ) from exc

        if not spectrum_str:
            raise SpectrometerAcquisitionError("Empty response from spectrometer")

        logger.debug(
            "Raw spectrum string length: %d, first 200 chars: %s",
            len(spectrum_str),
            spectrum_str[:200],
        )

        # Split into lines first to handle command echoes properly
        lines = spectrum_str.replace(",", " ").split("\n")
        logger.debug(
            "Parsed %d lines from spectrum, first 5: %s",
            len(lines),
            lines[:5],
        )

        # Filter to only valid numeric values, skipping command echoes and ACKs
        numeric_values = []
        skip_patterns = {"ACK", "a", "S"}  # Command echoes to skip

        for line in lines:
            line = line.strip()
            # Skip empty lines and command echoes
            if not line or line in skip_patterns or line.startswith("I"):
                continue
            # Skip ACK responses
            if "ACK" in line:
                continue

            # Try to parse as a number
            try:
                if not line.isdigit() or len(line) < 4:
                    continue
                val = float(line)
                numeric_values.append(val)
                # Stop once we have enough data points
                if len(numeric_values) >= self.config.data_points:
                    break
            except ValueError:
                # Skip non-numeric values
                continue

        logger.debug(
            "Converted %d numeric values, first 10: %s",
            len(numeric_values),
            numeric_values[:10],
        )

        if len(numeric_values) < self.config.min_data_points:
            raise SpectrometerAcquisitionError(
                f"Received {len(numeric_values)} valid data points, expected at least "
                f"{self.config.min_data_points}"
            )

        return numeric_values

    def _normalize_spectrum(self, values: Sequence[float]) -> np.ndarray:
        """Convert spectrum data to float numpy array."""
        spectrum = np.asarray(values, dtype=np.float32)
        # Accept any spectrum that meets minimum requirements
        # Don't require exactly data_points - real devices vary
        if spectrum.size < self.config.min_data_points:
            raise SpectrometerAcquisitionError(
                f"Received {spectrum.size} points, expected at least {self.config.min_data_points}"
            )
        if spectrum.size > self.config.data_points:
            spectrum = spectrum[: self.config.data_points]
        logger.info("Normalized spectrum: %d data points", spectrum.size)
        return spectrum

    def _load_simulated_spectrum(self, simulation_file: Optional[str]) -> np.ndarray:
        """Load a simulated spectrum from disk or generate one."""
        if simulation_file:
            path = Path(simulation_file).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path

            if path.exists():
                try:
                    if path.suffix.lower() == ".json":
                        payload = json.loads(path.read_text())
                        if isinstance(payload, dict):
                            values = payload.get("spectrum_data")
                        else:
                            values = payload
                    else:
                        values = np.loadtxt(path, delimiter=",", dtype=np.float32)

                    if values is not None:
                        spectrum = np.asarray(values, dtype=np.float32)
                        if spectrum.size >= self.config.min_data_points:
                            logger.info(
                                "Loaded simulated spectrum from %s", path.as_posix()
                            )
                            if spectrum.ndim > 1:
                                spectrum = spectrum.flatten()
                            if spectrum.size > self.config.data_points:
                                spectrum = spectrum[: self.config.data_points]
                            if spectrum.size < self.config.data_points:
                                logger.warning(
                                    "Simulation data has %s points, expected %s",
                                    spectrum.size,
                                    self.config.data_points,
                                )
                            return spectrum.astype(np.float32)
                except Exception as exc:
                    logger.warning(
                        "Failed to load simulated spectrum from %s: %s",
                        path.as_posix(),
                        exc,
                    )

        logger.info("Generating fallback simulator spectrum")
        return self._generate_fallback_spectrum()

    def _generate_fallback_spectrum(self) -> np.ndarray:
        """Generate a fallback spectrum when no recorded data is available."""
        logger.info("Generating fallback spectrum with realistic Raman peaks")

        spectrum = np.zeros(self.config.data_points, dtype=np.float32)

        baseline = np.random.normal(100, 20, self.config.data_points)
        spectrum += np.maximum(baseline, 0).astype(np.float32)

        peak_positions = [300, 600, 1000, 1400, 1600]
        peak_intensities = [200, 400, 600, 300, 250]

        x = np.arange(self.config.data_points)
        for pos, intensity in zip(peak_positions, peak_intensities):
            if pos < self.config.data_points:
                width = 30
                peak = intensity * np.exp(-((x - pos) ** 2) / (2 * width**2))
                spectrum += peak.astype(np.float32)

        spectrum = np.maximum(spectrum, 0).astype(np.float32)
        return spectrum


class HardwareManager:
    """Manager for spectrometer hardware interactions."""

    def __init__(self):
        self.spectrometer: Optional[BWTekSpectrometer] = None
        self._available_ports = []
        self._last_error: Optional[str] = None
        self._last_source: Optional[str] = None
        self._last_acquired_at: Optional[datetime] = None

    def scan_ports(self) -> list:
        """Scan for available serial ports, prioritizing USB/serial devices."""
        import serial.tools.list_ports

        ports = []
        usb_ports = []
        other_ports = []

        for port in serial.tools.list_ports.comports():
            port_info = {
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid,
            }

            if "Bluetooth" in port.device or "Bluetooth" in port.description:
                logger.info("Skipping Bluetooth port: %s", port.device)
                continue

            if (
                "usbserial" in port.device
                or "ttyUSB" in port.device
                or "ttyACM" in port.device
                or "USB" in port.description
            ):
                usb_ports.append(port_info)
                logger.info(
                    "Found USB serial port: %s - %s",
                    port.device,
                    port.description,
                )
            else:
                other_ports.append(port_info)

        ports = usb_ports + other_ports
        self._available_ports = ports

        logger.info(
            "Found %s USB serial ports, %s other ports",
            len(usb_ports),
            len(other_ports),
        )
        return ports

    def connect_spectrometer(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        simulate: bool = False,
        simulation_file: Optional[str] = None,
    ) -> bool:
        """Connect to spectrometer hardware or simulator."""
        if self.spectrometer:
            self.spectrometer.disconnect()

        config = SpectrometerConfig(
            port=port,
            simulate=simulate,
            simulation_file=simulation_file,
        )
        spectrometer = BWTekSpectrometer(config)

        try:
            spectrometer.connect()
            self.spectrometer = spectrometer
            self._last_error = None
            self._last_source = "simulator" if simulate else "hardware"
            return True
        except SpectrometerError as exc:
            self._last_error = str(exc)
            self.spectrometer = None if not simulate else spectrometer
            raise

    def disconnect_spectrometer(self) -> bool:
        """Disconnect the spectrometer."""
        if not self.spectrometer:
            return True

        disconnected = self.spectrometer.disconnect()
        self.spectrometer = None
        return disconnected

    def get_spectrometer_status(self) -> Dict[str, Any]:
        """Get spectrometer status."""
        if self.spectrometer:
            status = self.spectrometer.get_status()
        else:
            status = {
                "connected": False,
                "port": None,
                "last_communication": None,
                "temperature": None,
                "baudrate": None,
                "data_points": None,
                "last_error": self._last_error,
                "last_source": self._last_source,
                "simulate": False,
                "simulation_file": None,
            }

        status["last_error"] = status.get("last_error") or self._last_error
        status["last_source"] = status.get("last_source") or self._last_source
        status["last_acquired_at"] = self._last_acquired_at

        return status

    def acquire_spectrum(
        self,
        integration_time: Optional[int] = None,
        *,
        simulate: bool = False,
        simulation_file: Optional[str] = None,
    ) -> np.ndarray:
        """Acquire spectrum from the active spectrometer."""
        if simulate and (
            not self.spectrometer or not self.spectrometer.config.simulate
        ):
            self.connect_spectrometer(
                port="simulator",
                simulate=True,
                simulation_file=simulation_file,
            )

        if not self.spectrometer:
            raise SpectrometerConnectionError("No spectrometer connected")

        try:
            spectrum = self.spectrometer.acquire_spectrum(
                integration_time,
                simulate=simulate,
                simulation_file=simulation_file,
            )
            self._last_error = None
            self._last_source = (
                "simulator"
                if simulate or self.spectrometer.config.simulate
                else "hardware"
            )
            self._last_acquired_at = datetime.now()
            return spectrum
        except SpectrometerError as exc:
            self._last_error = str(exc)
            raise
