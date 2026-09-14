####
# Queue management for ActiveTigger
# This module manages the queue of tasks for CPU and GPU processing using Loky.
# There is no multi-GPU support in this version.
####

import asyncio
import datetime
import multiprocessing
import uuid
from datetime import timezone

# manage the executor
from loky import get_reusable_executor

from activetigger.datamodels import QueueStateTaskModel, QueueTaskModel
from activetigger.errors import ServerBusyError
from activetigger.tasks.base_task import BaseTask

multiprocessing.set_start_method("spawn", force=True)


class Queue:
    """
    Managining parallel processes for computation

    Use Loky as a backend for one common executor

    With 2 waiting queues to differentiate between CPU and GPU jobs
    (to limit concurrency in GPU memory usage)
    """

    max_processes: int = 20
    worker_idle_timeout_seconds: int = 60
    nb_workers: int
    nb_workers_cpu: int
    nb_workers_gpu: int
    current: list[QueueTaskModel]
    last_restart: datetime.datetime
    task_timeout_seconds: int

    TERMINAL_STATES = {"done", "cancelled", "failed"}

    def __init__(
        self,
        nb_workers_cpu: int = 5,
        nb_workers_gpu: int = 1,
        task_timeout_seconds: int = 4 * 3600,
    ) -> None:
        """
        Initiating the queue
        :param nb_workers_cpu: Number of CPU workers
        :param nb_workers_gpu: Number of GPU workers
        :param task_timeout_seconds: Per-task wall-clock budget. Once a running
            task exceeds this, its cancel event is set and it is marked
            "failed" so the dispatcher can free the slot. Loky may keep the
            underlying worker until the task returns, but the queue stops
            counting it against worker capacity.
        :return: None
        """
        self.nb_workers_cpu = nb_workers_cpu
        self.nb_workers_gpu = nb_workers_gpu
        self.nb_workers = nb_workers_cpu + nb_workers_gpu
        self.task_timeout_seconds = task_timeout_seconds
        self.current = []

        # manager for cross-process event signaling
        self.manager = multiprocessing.Manager()

        # create the executor
        self.executor = get_reusable_executor(
            max_workers=self.nb_workers, timeout=self.worker_idle_timeout_seconds
        )

        # launch a regular update on the queue; deferred when no loop is
        # running yet (e.g. module-import-time instantiation outside uvicorn),
        # ensure_update_task() picks it up later from the FastAPI lifespan.
        self.task: asyncio.Task | None = None
        self.ensure_update_task()

    def ensure_update_task(self) -> None:
        """
        Schedule the background update loop if it isn't already running.
        Safe to call repeatedly; required for callers that build a Queue
        outside a running event loop (tests, module-import-time init).
        """
        if self.task is not None and not self.task.done():
            return
        try:
            self.task = asyncio.create_task(self._update_queue(timeout=0.5))
        except RuntimeError:
            self.task = None

    def __del__(self) -> None:
        """
        Destructor to close the queue
        """
        if hasattr(self, "task"):
            if self.task is not None:
                self.task.cancel()
        if hasattr(self, "manager"):
            self.manager.shutdown()

    def _refresh_states(self) -> None:
        """
        Sync task.state with future status.
        """
        now = datetime.datetime.now(timezone.utc)
        for t in self.current:
            if t.state != "running" or t.future is None:
                continue
            if t.future.done():
                t.state = "failed" if t.future.exception() else "done"
                continue
            if t.running_since is None:
                continue
            elapsed = (now - t.running_since).total_seconds()
            if elapsed >= self.task_timeout_seconds:
                print(
                    f"Task {t.unique_id} ({t.kind}) exceeded timeout "
                    f"({elapsed:.0f}s >= {self.task_timeout_seconds}s); marking as failed.",
                    flush=True,
                )
                try:
                    if t.event is not None:
                        t.event.set()
                except Exception as e:
                    print(f"Failed to signal cancel event for {t.unique_id}: {e}", flush=True)
                t.state = "failed"
        # To see the queue for debug
        if len(self.current) > 0:
            print("Active processes", self.current)

    def _submit_task(self, t: QueueTaskModel, now: datetime.datetime) -> None:
        """
        Submit a task to the executor, rebuilding the executor if it is
        broken. When a worker dies (host OOM kill, CUDA crash), loky marks
        the executor broken and every submit raises; without a rebuild the
        task would stay "pending" and the queue would wedge forever.
        """
        try:
            t.future = self.executor.submit(t.task)
        except Exception as e:
            print(f"Executor unusable ({e}); rebuilding executor.", flush=True)
            # loky's reusable executor singleton returns a fresh executor
            # when the previous one is broken or shut down
            self.executor = get_reusable_executor(
                max_workers=self.nb_workers, timeout=self.worker_idle_timeout_seconds
            )
            t.future = self.executor.submit(t.task)
        t.state = "running"
        t.running_since = now

    def _dispatch_pending_tasks(self) -> None:
        """
        Check for pending tasks and submit them to the executor.
        Runs in a thread to avoid blocking the event loop
        (executor.submit may spawn worker processes).
        """
        # refresh state from futures so completed/timed-out tasks free their slots
        self._refresh_states()

        # active tasks in the queue
        nb_active_processes_gpu = len(
            [i for i in self.current if i.queue == "gpu" and i.state == "running"]
        )
        nb_active_processes_cpu = len(
            [i for i in self.current if i.queue == "cpu" and i.state == "running"]
        )

        # pending tasks in the queue
        task_gpu = [i for i in self.current if i.queue == "gpu" and i.state == "pending"]
        task_cpu = [i for i in self.current if i.queue == "cpu" and i.state == "pending"]

        now = datetime.datetime.now(timezone.utc)

        # a worker available and possible to have gpu
        if (
            nb_active_processes_gpu < self.nb_workers_gpu
            and (nb_active_processes_gpu + nb_active_processes_cpu) < self.nb_workers
            and len(task_gpu) > 0
        ):
            self._submit_task(task_gpu[0], now)

        # a worker available and possible to have cpu
        if (
            nb_active_processes_cpu < self.nb_workers_cpu
            and (nb_active_processes_gpu + nb_active_processes_cpu) < self.nb_workers
            and len(task_cpu) > 0
        ):
            self._submit_task(task_cpu[0], now)

    async def _update_queue(self, timeout: float = 1) -> None:
        """
        Update the queue every X seconds.
        Add new tasks to the executor if there are available workers.
        """
        while True:
            try:
                await asyncio.to_thread(self._dispatch_pending_tasks)
            except asyncio.CancelledError:
                print("Queue update task cancelled.")
                return
            except Exception as e:
                print(f"Error in queue dispatch: {e}")
            await asyncio.sleep(timeout)

    def add_task(self, kind: str, project_slug: str, task: BaseTask, queue: str = "cpu") -> str:
        """
        Add a task in the queue, first as pending in the current list
        """
        # test if the queue is not full
        if len(self.current) > self.max_processes:
            raise ServerBusyError("Queue is full. Wait for process to finish.")

        # generate a unique id
        unique_id = str(uuid.uuid4())
        task.unique_id = unique_id

        # set an event to inform the end of the process
        event = self.manager.Event()
        task.event = event

        # add it in the current processes
        self.current.append(
            QueueTaskModel(
                unique_id=unique_id,
                kind=kind,
                project_slug=project_slug,
                state="pending",
                future=None,
                event=event,
                starting_time=datetime.datetime.now(timezone.utc),
                queue=queue,
                task=task,
            )
        )

        return unique_id

    def get(self, unique_id: str) -> QueueTaskModel | None:
        """
        Get a running process
        """
        element = [i for i in self.current if i.unique_id == unique_id]
        if len(element) == 0:
            return None
        t = element[0]
        if t.state == "running" and t.future is not None and t.future.done():
            t.state = "failed" if t.future.exception() else "done"
        return t

    def kill(self, unique_id: str) -> None:
        """
        Send a kill process with the event manager.
        Idempotent: silently no-ops if the process is no longer in the queue
        (already finished, cleaned up, or never existed).
        """
        element = [i for i in self.current if i.unique_id == unique_id]
        if len(element) == 0:
            return
        element[0].event.set()
        element[0].state = "cancelled"

    def delete(self, ids: str | list) -> None:
        """
        Delete completed elements from the stack
        """
        if isinstance(ids, str):
            ids = [ids]
        for i in [t for t in self.current if t.unique_id in ids]:
            if i.state == "cancelled":
                print("Deleting a unfinished process", flush=True)
            self.current.remove(i)

    def state(self) -> list[QueueStateTaskModel]:
        """
        Return state of the queue
        """
        out: list[QueueStateTaskModel] = []
        for process in self.current:
            future = process.future
            future_done = future is not None and future.done()
            if future is not None and future_done:
                exc = future.exception()
                exception = str(exc) if exc else None
            else:
                exception = None
            if process.state == "running" and future_done:
                reported_state = "failed" if exception else "done"
            else:
                reported_state = process.state
            out.append(
                QueueStateTaskModel(
                    unique_id=process.unique_id,
                    state=reported_state,
                    exception=exception,
                    kind=process.kind,
                )
            )
        return out

    def get_nb_waiting_processes(self, queue: str = "cpu") -> int:
        """
        Number of waiting processes
        """
        return len([f for f in self.current if f.queue == queue and f.state == "pending"])

    def display_info(self, renew: int = 20) -> None:
        """
        Check if the exector still works
        if not, recreate it
        """
        print(self.state())
        print(
            "waiting",
            self.get_nb_waiting_processes("cpu"),
            self.get_nb_waiting_processes("gpu"),
        )
        return None

    def clean_old_processes(self, timeout: int = 2) -> None:
        """
        Remove old processes.

        Reaps tasks that are in a terminal state, plus any task whose total
        lifetime in the queue exceeds `timeout` hours regardless of state —
        otherwise a task wedged in "running" or "pending" would never be
        cleaned.
        """
        n = len(self.current)
        now = datetime.datetime.now(timezone.utc)
        old_processes_ids = []
        for i in self.current:
            age_hours = (now - i.starting_time).total_seconds() / 3600
            if i.state in self.TERMINAL_STATES:
                old_processes_ids.append(i.unique_id)
            elif age_hours >= timeout:
                print(
                    f"Reaping stuck task {i.unique_id} ({i.kind}) in state "
                    f"{i.state!r} after {age_hours:.1f}h.",
                    flush=True,
                )
                old_processes_ids.append(i.unique_id)
        if len(old_processes_ids) > 0:
            self.delete(old_processes_ids)
        if n != len(self.current):
            print(f"Cleaned {n - len(self.current)} processes")
        return None

    def restart(self) -> None:
        """
        Restart the queue: drop pending state, tear down the executor and
        the Manager, and rebuild both. Without rebuilding `self.executor`,
        every subsequent `executor.submit` fails with "cannot schedule new
        futures after shutdown" and the queue silently wedges.
        """
        self.current = []

        try:
            # kill_workers: terminate busy workers too — a wedged task would
            # otherwise survive the restart and keep its GPU memory forever
            self.executor.shutdown(wait=False, kill_workers=True)
        except Exception as e:
            print(f"Error shutting down old executor: {e}", flush=True)

        # The Manager backs the Events held by the tasks we just dropped;
        # rebuild it so newly-added tasks get fresh, live Events.
        try:
            self.manager.shutdown()
        except Exception as e:
            print(f"Error shutting down manager: {e}", flush=True)
        self.manager = multiprocessing.Manager()

        # loky's reusable executor is a process-global singleton; after
        # shutdown the next get_reusable_executor call returns a fresh one.
        self.executor = get_reusable_executor(
            max_workers=self.nb_workers, timeout=self.worker_idle_timeout_seconds
        )
