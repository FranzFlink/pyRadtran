# pyradtran/utils.py
"""
Utility helpers for pyRadtran.

Currently provides :class:`RadiosondeFinder`, which scans a directory tree
for radiosonde ``.dat`` files and returns the one closest in time to a
given target datetime.

.. autosummary::

   RadiosondeFinder

Notes
-----
The finder is what connects a local radiosonde archive to a simulation.
With

.. code-block:: yaml

    paths:
      radiosonde_base: /data/halo_ac3/radiosonde
    simulation_defaults:
      h2o_source: radiosonde

every simulated point gets the line ``radiosonde <closest file> H2O RH`` in
its ``uvspec`` input, i.e. the pressure, temperature and relative-humidity
profile of the sounding nearest in time replaces the lower part of the
background atmosphere.

See Also
--------
pyradtran.io.RadiosondeAtmosphereGenerator : Download soundings from IGRA
    and write them in the expected format.
pyradtran.era5.era5_atmosphere_file : Atmosphere profiles from ERA5.
"""

import logging
import re
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class RadiosondeFinder:
    r"""Locate the radiosonde file closest in time to a target datetime.

    On construction the *base_path* directory tree is scanned recursively
    for files named ``YYYYMMDD_SSSSSSOD.dat`` — the launch date followed
    by the five-digit seconds of day (UTC), e.g.
    ``20220328_43200SOD.dat`` for a sounding at 12:00 UTC on 28 March
    2022.  Subsequent calls to :meth:`find_closest` perform a binary
    search on the time-sorted file list.

    Parameters
    ----------
    base_path : pathlib.Path or None
        Root directory to scan.  If *None* (or not an existing directory),
        nothing is indexed and all look-ups return *None*.

    Attributes
    ----------
    base_path : pathlib.Path or None
        The directory given at construction.

    See Also
    --------
    pyradtran.io.RadiosondeAtmosphereGenerator : Fetch soundings from IGRA.
    pyradtran.config.PathsConfig : ``radiosonde_base`` is the directory
        handed to this class during a simulation.

    Notes
    -----
    A file name with date :math:`d` and seconds of day :math:`s` is
    assigned the time

    .. math::

        t = d\ 	ext{00:00 UTC} + s\ \mathrm{seconds},
        \qquad 0 \le s \le 99999 .

    Files that end in ``.dat`` but do not match the pattern are ignored,
    as are files whose date part is not a valid calendar date (a warning
    is logged for the latter).  Only the name is used; the file contents
    are never read here.  The index is built once — files added later are
    not seen until a new finder is created.

    The files themselves are plain three-column tables (pressure in hPa,
    temperature in K, relative humidity in %), ordered from the top of the
    sounding downwards, as ``uvspec``'s ``radiosonde ... H2O RH`` option
    expects.

    Examples
    --------
    A small archive with one directory per flight:

    >>> import tempfile
    >>> from datetime import datetime
    >>> from pathlib import Path
    >>> from pyradtran import RadiosondeFinder
    >>> root = Path(tempfile.mkdtemp())
    >>> for name in [
    ...     "Flight_20220328/20220328_37218SOD.dat",
    ...     "Flight_20220328/20220328_45046SOD.dat",
    ...     "Flight_20220329/20220329_40000SOD.dat",
    ...     "Flight_20220329/notes.dat",
    ... ]:
    ...     (root / name).parent.mkdir(exist_ok=True)
    ...     (root / name).touch()
    >>> finder = RadiosondeFinder(root)

    37218 s is 10:20 UTC and 45046 s is 12:31 UTC, so a measurement at
    11:00 UTC is matched to the first sounding and one at 12:00 UTC to
    the second:

    >>> finder.find_closest(datetime(2022, 3, 28, 11, 0)).name
    '20220328_37218SOD.dat'
    >>> finder.find_closest(datetime(2022, 3, 28, 12, 0)).name
    '20220328_45046SOD.dat'

    Times outside the covered period get the first or last sounding:

    >>> finder.find_closest(datetime(2023, 1, 1)).name
    '20220329_40000SOD.dat'

    Without a directory nothing is found:

    >>> RadiosondeFinder(None).find_closest(datetime(2022, 3, 28)) is None
    True
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
        """Index all ``YYYYMMDD_SSSSSSOD.dat`` files below *base_path*.

        Fills ``_sonde_data`` with ``(UTC datetime, path)`` tuples sorted
        by time.  Does nothing if *base_path* is not a directory.
        """
        if not self.base_path.is_dir():
            logger.debug(
                f"Radiosonde base path does not exist or not provided: {self.base_path}"
            )
            return

        logger.info(f"Scanning for radiosondes under: {self.base_path}")
        sonde_files = []
        for sonde_path in self.base_path.rglob("*.dat"):
            match = self._SONDE_FILENAME_PATTERN.search(sonde_path.name)
            if match:
                date_str, sod_str = match.groups()
                try:
                    # Assume sonde filenames are UTC
                    base_date = datetime.strptime(date_str, "%Y%m%d").replace(
                        tzinfo=timezone.utc
                    )
                    # SOD seems to be seconds of day
                    file_datetime = base_date + timedelta(seconds=int(sod_str))
                    sonde_files.append((file_datetime, sonde_path))
                except ValueError:
                    logger.warning(
                        f"Could not parse timestamp from sonde file: {sonde_path.name}"
                    )

        self._sonde_data = sorted(sonde_files, key=lambda item: item[0])
        logger.info(f"Found and parsed {len(self._sonde_data)} radiosonde files.")
        if not self._sonde_data:
            logger.warning("No valid radiosonde files found in the specified path.")

    def find_closest(
        self, target_dt: datetime, max_hours: Optional[float] = None
    ) -> Optional[Path]:
        r"""Return the radiosonde file closest in time to *target_dt*.

        Parameters
        ----------
        target_dt : datetime.datetime
            Target time.  A timezone-naive value is taken as UTC; an
            aware value is converted to UTC.
        max_hours : float, optional
            Largest accepted time difference in hours.  When the closest
            sounding is farther away, *None* is returned.  By default
            there is no limit.

        Returns
        -------
        pathlib.Path or None
            Path of the best-matching file (below *base_path*, absolute
            if *base_path* is), or *None* when no files have been indexed
            or the closest one is more than *max_hours* away.

        Notes
        -----
        The result minimises :math:`|t_i - t|` over all indexed sounding
        times :math:`t_i`.  Without *max_hours* a target before the first
        (after the last) sounding returns the first (last) file, however
        far away; a warning is logged whenever the returned sounding is
        more than 12 h from the target.  If the target lies exactly
        halfway between two soundings the earlier one is returned.

        Examples
        --------
        >>> import tempfile
        >>> from datetime import datetime, timedelta, timezone
        >>> from pathlib import Path
        >>> from pyradtran import RadiosondeFinder
        >>> root = Path(tempfile.mkdtemp())
        >>> for name in ["20220328_36000SOD.dat", "20220328_43200SOD.dat"]:
        ...     (root / name).touch()
        >>> finder = RadiosondeFinder(root)
        >>> finder.find_closest(datetime(2022, 3, 28, 10, 30)).name
        '20220328_36000SOD.dat'

        Exactly halfway (11:00 UTC between 10:00 and 12:00) the earlier
        sounding wins:

        >>> finder.find_closest(datetime(2022, 3, 28, 11, 0)).name
        '20220328_36000SOD.dat'

        Aware datetimes are converted — 13:30 at UTC+2 is 11:30 UTC:

        >>> cest = timezone(timedelta(hours=2))
        >>> finder.find_closest(datetime(2022, 3, 28, 13, 30, tzinfo=cest)).name
        '20220328_43200SOD.dat'

        A sounding months away is still returned (with a logged warning)
        unless a limit is given:

        >>> finder.find_closest(datetime(2022, 9, 1)).name
        '20220328_43200SOD.dat'
        >>> finder.find_closest(datetime(2022, 9, 1), max_hours=6) is None
        True
        """
        if not self._sonde_data:
            return None

        # Compare in UTC (a naive target_dt is assumed to be UTC)
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)
        else:
            target_dt = target_dt.astimezone(timezone.utc)

        # insertion point of target_dt in the time-sorted index
        pos = bisect_left([item[0] for item in self._sonde_data], target_dt)

        # Candidates: the neighbours of the insertion point (earlier one
        # first, so it wins a tie).
        candidates = self._sonde_data[max(pos - 1, 0) : pos + 1]
        sonde_dt, path = min(candidates, key=lambda item: abs(item[0] - target_dt))

        hours = abs(sonde_dt - target_dt).total_seconds() / 3600
        if max_hours is not None and hours > max_hours:
            logger.warning(
                f"Closest radiosonde {path.name} is {hours:.1f} h from "
                f"{target_dt:%Y-%m-%d %H:%M} UTC (max_hours={max_hours}); none used."
            )
            return None
        if hours > 12:
            logger.warning(
                f"Closest radiosonde {path.name} is {hours:.1f} h from "
                f"{target_dt:%Y-%m-%d %H:%M} UTC."
            )
        return path

    def find_radiosonde_file(
        self, dt: datetime, latitude: float, longitude: float
    ) -> Optional[Path]:
        """Find the radiosonde file closest in time to *dt*.

        This is the method the simulation engine calls for every point
        when ``simulation_defaults.h2o_source`` is ``"radiosonde"``.  It
        currently delegates to :meth:`find_closest`; the position is
        ignored.

        Parameters
        ----------
        dt : datetime.datetime
            Target time (UTC if timezone-naive).
        latitude, longitude : float
            Reserved for future spatial matching; currently unused.

        Returns
        -------
        pathlib.Path or None
            Closest file in time, or *None* when no files are indexed.

        See Also
        --------
        find_closest : The underlying time search.

        Examples
        --------
        >>> finder = RadiosondeFinder(Path("/data/halo_ac3/radiosonde"))  # doctest: +SKIP
        >>> path = finder.find_radiosonde_file(  # doctest: +SKIP
        ...     datetime(2022, 3, 28, 11, 0), latitude=78.9, longitude=11.9
        ... )
        """
        return self.find_closest(dt)
