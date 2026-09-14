import threading
from datetime import datetime, timedelta, timezone

import pandas as pd

from activetigger.datamodels import (
    EventsModel,
    MonitoringActivityModel,
    MonitoringActivityPointModel,
    MonitoringEmissionsModel,
    MonitoringGpuModel,
    MonitoringLanguageModelsModel,
    MonitoringMetricsModel,
    MonitoringQuickModelsModel,
)
from activetigger.db.manager import DatabaseManager
from activetigger.errors import NotFoundError


class GpuMonitor:
    """Background poller that tracks peak GPU memory used (in GB) during a task.

    Used memory is `total_memory - available_memory` measured via
    `activetigger.functions.get_gpu_memory_info`. If no GPU is available on the
    first sample, the polling thread is not started.
    """

    def __init__(self, poll_interval: float = 5.0) -> None:
        self._poll_interval = poll_interval
        self._max_used_gb = 0.0
        self._total_memory_gb = 0.0
        self._gpu_available = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._max_used_gb = 0.0
        self._total_memory_gb = 0.0
        self._gpu_available = False
        # Synchronous first sample: skip the polling thread when no GPU is present.
        self._sample()
        if not self._gpu_available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self._poll_interval * 2)
            self._thread = None
        # Final sample to catch a peak that occurred near task end.
        self._sample()
        return self._max_used_gb

    def _sample(self) -> None:
        # Local import to avoid importing torch at module load time.
        from activetigger.functions import get_gpu_memory_info

        try:
            info = get_gpu_memory_info()
        except Exception:
            return
        self._gpu_available = bool(info.gpu_available)
        if not info.gpu_available:
            return
        self._total_memory_gb = float(info.total_memory)
        used = float(info.total_memory) - float(info.available_memory)
        if used > self._max_used_gb:
            self._max_used_gb = used

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self._poll_interval)

    @property
    def max_used_gb(self) -> float:
        return round(self._max_used_gb, 3)

    @property
    def total_memory_gb(self) -> float:
        return round(self._total_memory_gb, 3)

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available


class EmissionsMonitor:
    """Wraps a codecarbon OfflineEmissionsTracker for the lifetime of a task.

    Provides safe start/stop and exposes the final emissions in kg CO2eq plus
    energy consumed in kWh. All failure modes are swallowed so that emission
    tracking can never crash the host task.
    """

    def __init__(
        self,
        enabled: bool = True,
        country_iso_code: str | None = None,
        measure_power_secs: int = 15,
    ) -> None:
        self._enabled = enabled
        self._country = country_iso_code
        self._measure_power_secs = measure_power_secs
        self._tracker = None
        self._available = False
        self._emissions_kg = 0.0
        self._energy_kwh = 0.0
        self._country_iso_code: str | None = None

    def start(self) -> None:
        if not self._enabled:
            print("[EmissionsMonitor] disabled by config (carbon_enabled=false); skipping")
            return
        if self._tracker is not None:
            return
        try:
            from codecarbon import OfflineEmissionsTracker  # local import: optional dep
        except Exception as e:
            print(f"[EmissionsMonitor] codecarbon not importable: {type(e).__name__}: {e}")
            return
        try:
            kwargs: dict = {
                "save_to_file": False,
                "log_level": "error",
                "tracking_mode": "process",
                "allow_multiple_runs": True,
                "measure_power_secs": self._measure_power_secs,
            }
            if self._country:
                kwargs["country_iso_code"] = self._country
            else:
                # OfflineEmissionsTracker requires *some* location; default to
                # France when no admin override is set. Online geolocation would
                # require network at task start, which we want to avoid.
                kwargs["country_iso_code"] = "FRA"
            self._tracker = OfflineEmissionsTracker(**kwargs)
            self._tracker.start()
            self._available = True
        except Exception:
            print("[EmissionsMonitor] failed to start tracker ")
            self._tracker = None
            self._available = False

    def stop(self) -> float:
        if self._tracker is None:
            return self._emissions_kg
        try:
            emissions = self._tracker.stop()
            if emissions is not None:
                self._emissions_kg = float(emissions)
            data = getattr(self._tracker, "final_emissions_data", None)
            if data is not None:
                self._energy_kwh = float(getattr(data, "energy_consumed", 0.0) or 0.0)
                self._country_iso_code = getattr(data, "country_iso_code", None)
            else:
                print(
                    "[EmissionsMonitor] tracker stopped but final_emissions_data missing; "
                    "energy_kwh/country left at defaults"
                )
        except Exception as e:
            print(f"[EmissionsMonitor] failed to stop tracker: {type(e).__name__}: {e}")
        finally:
            self._tracker = None
        return self._emissions_kg

    @property
    def available(self) -> bool:
        return self._available

    @property
    def emissions_kg(self) -> float:
        # 6 decimals: typical small-run emissions are ~1e-6 kg.
        return round(self._emissions_kg, 6)

    @property
    def energy_kwh(self) -> float:
        return round(self._energy_kwh, 6)

    @property
    def country_iso_code(self) -> str | None:
        return self._country_iso_code


class TaskTimer:
    """This object centralises the timing component in order to save them as part
    of the "additional_event" in the Monitoring.close_process function.

    Also wraps a `GpuMonitor` so every TaskTimer-using task automatically records
    the peak GPU memory used during its execution under the "gpu" event key, and
    an `EmissionsMonitor` recording carbon footprint under the "emissions" key.
    """

    body = {"start": "FAILED", "end": "FAILED", "duration": "FAILED", "order": None}

    def __init__(
        self,
        compulsory_steps: list[str],
        optional_steps: list[str] | None = None,
        gpu_poll_interval: float = 5.0,
    ) -> None:
        self.__additional_events = {step: self.body for step in compulsory_steps}
        self.__starts: dict[str, datetime] = {}
        self.__stops: list[str] = []
        self.__optional_steps: list[str] = optional_steps if optional_steps is not None else []
        self.__gpu_monitor = GpuMonitor(poll_interval=gpu_poll_interval)
        self.__gpu_monitor.start()
        # Local import to keep the config module out of monitoring's import path
        # for unit tests that build TaskTimer directly.
        from activetigger.config import config

        self.__emissions_monitor = EmissionsMonitor(
            enabled=bool(getattr(config, "carbon_enabled", True)),
            country_iso_code=getattr(config, "carbon_country", None),
        )
        self.__emissions_monitor.start()

    def start(self, step: str) -> None:
        """
        Starts the corresponding timer. Make sure that the step exists, if
        optional, initiate the step body.
        """

        if step in self.__optional_steps:
            self.__additional_events[step] = self.body
        if step not in self.__additional_events:
            raise Exception(
                (
                    f"TaskTimer.start(step): {step} is not one of the compulsory "
                    f"steps ({self.__additional_events.keys()})"
                )
            )
        if step in self.__starts:
            raise Exception((f"TaskTimer.start(step): {step} timer has already been started."))
        self.__starts[step] = datetime.now(timezone.utc)

    def stop(self, step: str) -> None:
        """
        Stops the timer
        """

        if step not in self.__starts:
            raise Exception(
                (
                    f"TaskTimer.stop(step): the step {step} timer was not started "
                    f"or previously stopped."
                )
            )
        if step in self.__stops:
            raise Exception(
                (f"TaskTimer.stop(step): the step {step} timer has already been stopped.")
            )

        end = datetime.now(timezone.utc)
        self.__stops += [str(step)]
        self.__additional_events[step] = {
            "start": self.__starts[step].isoformat(),
            "end": end.isoformat(),
            "duration": str((end - self.__starts[step]).total_seconds()),
            "order": str(len(self.__stops)),
        }

    def get_events(self) -> dict[str, dict[str, str | None]]:
        # Finalize GPU monitoring once, on the first call.
        if "gpu" not in self.__additional_events:
            self.__gpu_monitor.stop()
            self.__additional_events["gpu"] = {
                "available": "true" if self.__gpu_monitor.gpu_available else "false",
                "max_used_gb": str(self.__gpu_monitor.max_used_gb),
                "total_memory_gb": str(self.__gpu_monitor.total_memory_gb),
            }
        if "emissions" not in self.__additional_events:
            self.__emissions_monitor.stop()
            self.__additional_events["emissions"] = {
                "available": "true" if self.__emissions_monitor.available else "false",
                "emissions_kg": str(self.__emissions_monitor.emissions_kg),
                "energy_kwh": str(self.__emissions_monitor.energy_kwh),
                "country_iso_code": self.__emissions_monitor.country_iso_code or "",
            }
        return self.__additional_events


class Monitoring:
    """
    Manage messages on the interface
    - user messages
    - mail messages
    """

    db_manager: DatabaseManager
    project_slug: str | None
    ACTIVITY_CACHE_TTL_SECONDS: int = 300

    def __init__(self, db_manager: DatabaseManager, project_slug: str | None = None) -> None:
        self.db_manager = db_manager
        self.project_slug = project_slug
        self._activity_cache: dict[int, tuple[datetime, "MonitoringActivityModel"]] = {}
        self._activity_lock = threading.Lock()

    def register_process(
        self, process_name: str, kind: str, parameters: dict, user_name: str
    ) -> None:
        """
        Start a new monitored process
        """
        events = {"global": {"start": datetime.now(timezone.utc).isoformat()}}
        self.db_manager.monitoring_service.add_process(
            process_name=process_name,
            kind=kind,
            parameters=parameters,
            events=events,
            project_slug=self.project_slug,
            user_name=user_name,
        )

    def close_process(self, process_name: str, list_events: EventsModel) -> None:
        """
        Close a monitored process
        """
        start_entry = self.db_manager.monitoring_service.get_element_by_process(process_name)
        if start_entry is None:
            raise NotFoundError(f"Process {process_name} not found")

        # Save the duration of the global process
        events = start_entry.events
        end = datetime.now(timezone.utc)
        events["global"]["end"] = end.isoformat()
        # Ensure start_entry.time is timezone-aware for subtraction
        # (PostgreSQL returns tz-aware datetimes, SQLite returns naive ones)
        start_time = start_entry.time
        if start_time.tzinfo != end.tzinfo:
            start_time = start_time.astimezone(end.tzinfo)
        duration = (end - start_time).total_seconds()
        events["global"]["duration"] = duration
        events["global"]["order"] = -1

        # Add additional events to the events object
        additional_events = (
            list_events.events if list_events is not None and len(list_events.events) > 0 else None
        )
        if isinstance(additional_events, dict):
            # If additional events are passed on
            if ("start" in additional_events.keys()) or ("end" in additional_events.keys()):
                # We do not want to overwrite the "global"; keys, so we remove them prior to merging
                additional_events = {
                    key: value for key, value in additional_events.items() if key != "global"
                }
            # Merge the events with the additional events
            events.update(additional_events)

        self.db_manager.monitoring_service.update_process(
            process_name=process_name,
            events=events,
            duration=duration,
        )
        print(f"Process {process_name} closed in {duration} seconds")

    def get_completed_processes(
        self, kind: str, username: str | None, limit: int = 100
    ) -> pd.DataFrame:
        """
        Get completed processes of a given kind and user
        """
        processes = self.db_manager.monitoring_service.get_completed_processes(
            kind=kind, username=username, limit=limit
        )
        data = [
            {
                "process_name": process.process_name,
                "kind": process.kind,
                "time": process.time,
                "parameters": process.parameters,
                "events": process.events,
                "project_slug": process.project_slug,
                "user_name": process.user_name,
                "duration": process.duration,
                "username": process.user_name,
            }
            for process in processes
        ]
        df = pd.DataFrame(data)
        return df

    def get_failed_processes(self):
        raise NotImplementedError("Not implemented yet")

    def get_running_processes(self):
        raise NotImplementedError("Not implemented yet")

    def get_data(self, kind: str, state: str = "completed", limit: int = 100) -> pd.DataFrame:
        """
        Get monitoring data
        """
        if state == "completed":
            df_processes = self.get_completed_processes(kind=kind, username=None, limit=limit)
        else:
            raise NotImplementedError("Not implemented yet")
        return df_processes

    def get_metrics(self) -> MonitoringMetricsModel:
        """
        Get monitoring metrics
        """
        df_quickmodels = self.get_completed_processes(
            kind="train_quickmodel", username=None, limit=100
        )
        m_quickmodels = MonitoringQuickModelsModel(
            n=len(df_quickmodels),
            mean=0 if len(df_quickmodels) == 0 else df_quickmodels["duration"].mean(),
            std=0 if len(df_quickmodels) < 2 else df_quickmodels["duration"].std(),
        )
        df_languagemodels = self.get_completed_processes(
            kind="train_languagemodel", username=None, limit=100
        )
        m_languagemodels = MonitoringLanguageModelsModel(
            n=len(df_languagemodels),
            mean=0 if len(df_languagemodels) == 0 else df_languagemodels["duration"].mean(),
            std=0 if len(df_languagemodels) < 2 else df_languagemodels["duration"].std(),
        )
        m_gpu = self._compute_gpu_gbs_metrics(limit=100)
        m_emissions = self._compute_emissions_metrics(limit=100)
        return MonitoringMetricsModel(
            quickmodels=m_quickmodels,
            languagemodels=m_languagemodels,
            gpu=m_gpu,
            emissions=m_emissions,
        )

    def _compute_gpu_gbs_metrics(self, limit: int = 100) -> MonitoringGpuModel:
        """
        GPU usage per process in GB·s = peak GB (from events["gpu"]) * duration s.
        Only processes that actually used the GPU (peak > 0) are counted.
        """
        df_all = self.get_completed_processes(kind="all", username=None, limit=limit)
        if len(df_all) == 0:
            return MonitoringGpuModel(n=0, mean=0.0, std=0.0)

        def gbs(row) -> float:
            events = row.get("events") or {}
            gpu = events.get("gpu") if isinstance(events, dict) else None
            try:
                peak_gb = float((gpu or {}).get("max_used_gb", 0.0))
            except (TypeError, ValueError):
                peak_gb = 0.0
            try:
                duration = float(row.get("duration") or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            return peak_gb * duration

        gpu_series = df_all.apply(gbs, axis=1)
        gpu_series = gpu_series[gpu_series > 0]
        n = int(len(gpu_series))
        if n == 0:
            return MonitoringGpuModel(n=0, mean=0.0, std=0.0)
        return MonitoringGpuModel(
            n=n,
            mean=float(gpu_series.mean()),
            std=0.0 if n < 2 else float(gpu_series.std()),
        )

    def _compute_emissions_metrics(self, limit: int = 100) -> MonitoringEmissionsModel:
        """
        Carbon emissions per process in kg CO2eq (from events["emissions"]).
        Only processes that produced a positive measurement are counted.
        Returns mean, std, and the cumulative total across the window.
        """
        df_all = self.get_completed_processes(kind="all", username=None, limit=limit)
        if len(df_all) == 0:
            return MonitoringEmissionsModel(n=0, mean=0.0, std=0.0, total=0.0)

        def emissions_kg(row) -> float:
            events = row.get("events") or {}
            entry = events.get("emissions") if isinstance(events, dict) else None
            try:
                return float((entry or {}).get("emissions_kg", 0.0))
            except (TypeError, ValueError):
                return 0.0

        series = df_all.apply(emissions_kg, axis=1)
        positive = series[series > 0]
        n = int(len(positive))
        if n == 0:
            return MonitoringEmissionsModel(n=0, mean=0.0, std=0.0, total=0.0)
        return MonitoringEmissionsModel(
            n=n,
            mean=float(positive.mean()),
            std=0.0 if n < 2 else float(positive.std()),
            total=float(positive.sum()),
        )

    def get_weekly_activity(
        self, days: int = 7, force_refresh: bool = False
    ) -> MonitoringActivityModel:
        """
        Hourly timeline of annotations count and distinct active users
        over the last `days` days, cached for ACTIVITY_CACHE_TTL_SECONDS.

        Cached per `days` value on this Monitoring singleton; concurrent
        admin requests share the same in-flight computation via a lock.
        """
        now = datetime.now(timezone.utc)
        if not force_refresh:
            cached = self._activity_cache.get(days)
            if (
                cached is not None
                and (now - cached[0]).total_seconds() < self.ACTIVITY_CACHE_TTL_SECONDS
            ):
                return cached[1]

        with self._activity_lock:
            if not force_refresh:
                cached = self._activity_cache.get(days)
                if (
                    cached is not None
                    and (datetime.now(timezone.utc) - cached[0]).total_seconds()
                    < self.ACTIVITY_CACHE_TTL_SECONDS
                ):
                    return cached[1]
            value = self._compute_weekly_activity(days)
            self._activity_cache[days] = (datetime.now(timezone.utc), value)
            return value

    def _compute_weekly_activity(self, days: int) -> MonitoringActivityModel:
        """
        Build the fixed-length hourly series from SQL-aggregated buckets so that
        empty hours render as zero on the frontend.
        """
        annotations_by_hour, users_by_hour = (
            self.db_manager.monitoring_service.get_hourly_activity_counts(days=days)
        )

        total_hours = days * 24
        now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_hour = now_hour - timedelta(hours=total_hours - 1)

        activity = [
            MonitoringActivityPointModel(
                hour=(start_hour + timedelta(hours=i)).isoformat(),
                annotations=annotations_by_hour.get(start_hour + timedelta(hours=i), 0),
                active_users=users_by_hour.get(start_hour + timedelta(hours=i), 0),
            )
            for i in range(total_hours)
        ]
        return MonitoringActivityModel(activity=activity)
