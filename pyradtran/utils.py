# libradpy/utils.py
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from bisect import bisect_left
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class RadiosondeFinder:
    """
    Scans for radiosonde files and finds the closest one in time.
    """
    _SONDE_FILENAME_PATTERN = re.compile(r"(\d{8})_(\d{5})SOD\.dat")

    def __init__(self, base_path: Optional[Path]):
        self.base_path = base_path
        self._sonde_data: List[Tuple[datetime, Path]] = []
        if self.base_path:
            self._scan_sondes()
        else:
            logger.info("No radiosonde base path provided, skipping sonde scan.")

    def _scan_sondes(self):
        """Scans the base path for valid radiosonde files."""
        if not self.base_path or not self.base_path.is_dir():
            logger.warning(f"Radiosonde base path does not exist or not provided: {self.base_path}")
            return

        logger.info(f"Scanning for radiosondes under: {self.base_path}")
        sonde_files = []
        for sonde_path in self.base_path.rglob("*.dat"):
            match = self._SONDE_FILENAME_PATTERN.search(sonde_path.name)
            if match:
                date_str, sod_str = match.groups()
                try:
                    # Assume sonde filenames are UTC
                    base_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                    # SOD seems to be seconds of day
                    file_datetime = base_date + timedelta(seconds=int(sod_str))
                    sonde_files.append((file_datetime, sonde_path))
                except ValueError:
                    logger.warning(f"Could not parse timestamp from sonde file: {sonde_path.name}")

        self._sonde_data = sorted(sonde_files, key=lambda item: item[0])
        logger.info(f"Found and parsed {len(self._sonde_data)} radiosonde files.")
        if not self._sonde_data:
            logger.warning("No valid radiosonde files found in the specified path.")

    def find_closest(self, target_dt: datetime) -> Optional[Path]:
        """Finds the radiosonde file with the timestamp closest to the target datetime."""
        if not self._sonde_data:
            return None

        # Ensure target_dt is timezone-aware (assume UTC if naive)
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)
        elif target_dt.tzinfo != timezone.utc:
             # Convert to UTC if it's a different timezone
             target_dt = target_dt.astimezone(timezone.utc)


        sonde_times = [item[0] for item in self._sonde_data]

        # bisect_left finds the insertion point for target_dt in the sorted list sonde_times
        pos = bisect_left(sonde_times, target_dt)

        if pos == 0:
            # Target time is before the first sonde
            return self._sonde_data[0][1]
        if pos == len(sonde_times):
            # Target time is after the last sonde
            return self._sonde_data[-1][1]

        # Target time is between sonde_times[pos-1] and sonde_times[pos]
        dt_before = target_dt - sonde_times[pos - 1]
        dt_after = sonde_times[pos] - target_dt

        # Return the path of the sonde with the smaller time difference
        if dt_before <= dt_after:
            return self._sonde_data[pos - 1][1]
        else:
            return self._sonde_data[pos][1]

    def find_radiosonde_file(self, dt: datetime, latitude: float, longitude: float) -> Optional[Path]:
        """
        Finds the radiosonde file closest to the given datetime.
        Note: latitude and longitude are currently not used for spatial matching,
        only temporal matching is performed.
        """
        return self.find_closest(dt)

# Add other general utility functions here if needed