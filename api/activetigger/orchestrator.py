"""
Define the orchestrator and launch the instance that will span projects
Each project share a few elements:
- database manager
- queue
- users management
- messages management
"""

import asyncio
import os
import shutil
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
import psutil
import pyarrow.parquet as pq
from jose import jwt

from activetigger import __version__
from activetigger.config import config
from activetigger.datamodels import DatasetModel, LMComputing, ProjectBaseModel, ServerStateModel
from activetigger.db import DBException
from activetigger.db.manager import DatabaseManager
from activetigger.functions import (
    get_dir_size,
    get_gpu_memory_info,
    sanitize_uploaded_filename,
    slugify,
)
from activetigger.messages import Messages
from activetigger.monitoring import Monitoring
from activetigger.project import Project
from activetigger.queue_manager import Queue
from activetigger.tasks.duplicate_project import DuplicateProject
from activetigger.toolbox import Toolbox
from activetigger.uploads import get_upload_staging
from activetigger.users import Users


class Orchestrator:
    """
    Process to manage projects
    Use the config object to get parameters
    """

    starting_time: float
    db_name: str
    jwt_algorithm: str
    n_workers_cpu: int
    n_workers_gpu: int
    path: Path
    path_models: Path
    projects: dict[str, Project]
    db_manager: DatabaseManager
    queue: Queue
    users: Users
    messages: Messages
    max_projects: int
    project_creation_ongoing: dict[str, Project]
    monitoring: Monitoring
    toolbox: Toolbox

    def __init__(self) -> None:
        """
        Start the server
        """
        self.starting_time = time.time()

        self.max_projects = config.max_loaded_projects
        self.jwt_algorithm = config.jwt_algorithm
        self.n_workers_cpu = config.n_workers_cpu
        self.n_workers_gpu = config.n_workers_gpu

        # Define path
        self.path: Path = Path(config.data_path) / "projects"
        self.path_models: Path = Path(config.data_path) / "models"
        self.path_toy_datasets: Path = Path(self.path) / "toy-datasets"

        # create directories parent/static/models
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path.joinpath("static")).mkdir(parents=True, exist_ok=True)
            self.path_models.mkdir(exist_ok=True)
            self.path_toy_datasets.mkdir(exist_ok=True)
        except Exception as e:
            print(e)
            raise PermissionError(
                f"Cannot create directory: {e}. "
                f"Please ensure the data directory '{config.data_path}' is writable "
                f"by the current user (uid={os.getuid()}, gid={os.getgid()}). "
                f"Fix with: chown -R {os.getuid()}:{os.getgid()} {config.data_path}"
            ) from e

        # attributes of the server
        self.db_manager = DatabaseManager()
        self.queue = Queue(
            nb_workers_cpu=self.n_workers_cpu,
            nb_workers_gpu=self.n_workers_gpu,
        )
        self.messages = Messages(self.db_manager)
        self.users = Users(self.db_manager, self.messages)
        self.monitoring = Monitoring(self.db_manager)
        self.toolbox = Toolbox(self.queue)

        # projects in memory
        self.projects = {}

        # projects in creation
        self.project_creation_ongoing = {}

        # projects being duplicated (filesystem copy in the queue + pending
        # DB clone in the main process).
        self.project_duplication_ongoing: dict[str, dict[str, Any]] = {}

        # store creation errors so they survive project cleanup
        self.creation_errors: dict[str, str] = {}

        # cached heavy system stats (refreshed every _heavy_stats_interval seconds)
        self._heavy_stats_interval = 30
        self._heavy_stats_last_update = 0.0
        self._heavy_stats_cache: dict = self._collect_heavy_stats()

        # update the projects asynchronously; if no loop is running yet
        # (module-import-time init from main.py), defer to ensure_update_task()
        # which the FastAPI lifespan will call once the loop is up.
        self._running = True
        self._update_task: asyncio.Task | None = None
        self.ensure_update_task()

        # create the demo project if not existing at startup
        try:
            if "demo" not in self.existing_projects():
                self.create_demo_project("demo", "demo")
        except Exception as e:
            print(f"Error while creating demo project: {e}")
        self.server_state = self.get_server_state()

    def ensure_update_task(self) -> None:
        """
        Schedule the background _update loop if it isn't already running.
        Safe to call repeatedly. Used by the FastAPI lifespan to (re)start
        the loop when the orchestrator was instantiated outside one
        (module-import-time init, tests).
        """
        self.queue.ensure_update_task()
        if self._update_task is not None and not self._update_task.done():
            return
        try:
            self._update_task = asyncio.create_task(self._update(timeout=config.update_timeout))
        except RuntimeError:
            self._update_task = None

    def __del__(self):
        """
        Close the server
        """
        print("Ending the server")
        if self._update_task:
            self._update_task.cancel()
        self._running = False
        del self.queue
        print("Server off")

    def reset(self):
        """
        Erase the waiting queue and projects in memory
        """
        self.queue.restart()
        self.projects = {}
        self.project_creation_ongoing = {}
        self.project_duplication_ongoing = {}

    def starting_project_creation(self, project: ProjectBaseModel, username: str) -> str:
        """
        Start the project creation
        """
        project_slug = self.check_project_name(project.project_name)
        if project_slug in ["new", "logs"]:
            raise Exception("This project name is not valid - reserved word")
        # the upload endpoint sanitized the filename on disk; resolve the
        # client-provided name to the same form or the task won't find the file
        if project.filename is not None:
            project.filename = sanitize_uploaded_filename(project.filename)
        self.project_creation_ongoing[project_slug] = Project(
            project_slug,
            self.queue,
            self.db_manager,
            path_models=self.path_models,
            users=self.users,
            messages=self.messages,
        )
        # dispatch on kind before enqueuing (see docs/image-projects-strategy.md)
        if getattr(project, "kind", "text") == "image":
            self.project_creation_ongoing[project_slug].start_project_creation_imagexp(
                params=project,
                username=username,
                path=self.path,
            )
        else:
            self.project_creation_ongoing[project_slug].start_project_creation(
                params=project,
                username=username,
                path=self.path,
            )
        return project_slug

    def _sync_update_processes(self, project_lifetime: int) -> None:
        """
        Synchronous work for the update loop (runs in a thread to avoid
        blocking the async event loop).
        """
        timer = time.time()

        # Run project update_processes BEFORE clean_old_processes: the queue's
        # _refresh_states transitions completed futures to state="done", and
        # clean_old_processes reaps anything in TERMINAL_STATES — so reaping
        # first would drop tasks before finish_project_creation gets a chance
        # to consume the result, leaving creating projects stuck forever.
        to_del = []
        for p, project in list(self.projects.items()):
            try:
                if (timer - project.starting_time) > project_lifetime:
                    # Never evict a project with tasks in flight
                    if project.computing:
                        project.starting_time = timer
                    else:
                        to_del.append(p)
                        continue
                project.update_processes()
            except Exception as e:
                print(f"Error while updating project {p}: {e}")
                project.errors.add(f"Error while updating processes: {e}")
        for p in to_del:
            del self.projects[p]

        # loop on creating projects
        to_del = []
        for p, project in list(self.project_creation_ongoing.items()):
            try:
                project.update_processes()
                if project.status == "created":
                    to_del.append(p)
                if project.status == "error":
                    errors = project.errors.state()
                    if errors:
                        self.creation_errors[p] = errors[-1][0]
                    else:
                        self.creation_errors[p] = "Unknown error during project creation"
                    self.clean_unfinished_project(project_slug=p)
                    to_del.append(p)
            except Exception as e:
                print(f"Error while updating project {p}: {e}")
                self.creation_errors[p] = str(e)
                self.clean_unfinished_project(project_slug=p)
                to_del.append(p)
        for p in to_del:
            self.project_creation_ongoing.pop(p, None)

        # finalize in-flight project duplications (copy task → DB clone)
        try:
            self._sync_finish_duplications()
        except Exception as e:
            print(f"Error while finishing duplications: {e}")

        # Reap terminal/stuck tasks last, after results have been consumed.
        self.queue.clean_old_processes(timeout=4)

        # drop abandoned chunked-upload staging sessions
        try:
            get_upload_staging().clean_old()
        except Exception as e:
            print(f"Error while cleaning staged uploads: {e}")

    def _collect_heavy_stats(self) -> dict:
        """
        Collect expensive system metrics (CPU, memory, disk, GPU).
        Designed to run in a background thread.
        """
        cpu = psutil.cpu_percent()
        cpu_count = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        # measure the filesystem holding the data (external volume in Docker),
        # not the container root
        disk_info = psutil.disk_usage(config.data_path)
        at_memory = get_dir_size(config.data_path + "/projects")
        gpu = get_gpu_memory_info()
        return {
            "cpu": {"proportion": cpu, "total": cpu_count},
            "memory": {
                "proportion": memory_info.percent,
                "total": memory_info.total / (1024**3),
                "available": memory_info.available / (1024**3),
            },
            "disk": {
                "activetigger": at_memory,
                "proportion": disk_info.percent,
                "total": disk_info.total / (1024**3),
            },
            "gpu": gpu,
        }

    async def _update(self, timeout: int = 1, project_lifetime: int = 7200) -> None:
        """
        Update each project in memory every X seconds.
        - loaded project
        - project in creation
        Remove projects that are older than project_lifetime seconds.
        Heavy system stats are refreshed every _heavy_stats_interval seconds.
        All synchronous work runs in a thread to avoid blocking the event loop.
        """
        try:
            while self._running:
                try:
                    # run synchronous update_processes in a thread
                    await asyncio.to_thread(self._sync_update_processes, project_lifetime)

                    # refresh heavy system stats periodically in a thread
                    now = time.time()
                    if (now - self._heavy_stats_last_update) >= self._heavy_stats_interval:
                        self._heavy_stats_cache = await asyncio.to_thread(self._collect_heavy_stats)
                        self._heavy_stats_last_update = now

                    # build server state (cheap parts only, heavy stats come from cache)
                    self.server_state = self.get_server_state()
                except Exception as e:
                    print(f"Error in update loop iteration: {e}")
                    traceback.print_exc()

                await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            print("Update task cancelled.")
        finally:
            print("Update task finished.")

    def log_action(
        self,
        user: str,
        action: str,
        project: str = "general",
        connect="not implemented",
    ) -> None:
        """
        Log action in the database
        """
        self.db_manager.logs_service.add_log(user, action, project, connect)

    def get_logs(self, project_slug: str, limit: int, partial: bool = True) -> pd.DataFrame:
        """
        Get logs for a user/project
        project_slug: project slug or "all"
        """
        logs = self.db_manager.logs_service.get_logs("all", project_slug, limit)
        df = pd.DataFrame(logs, columns=["id", "time", "user", "project", "action", "NA"])
        if partial:
            return df[~df["action"].str.contains("INFO ")]
        return df

    def get_server_state(self) -> ServerStateModel:
        """
        Build server state.
        Light data (active projects, queue) is computed inline.
        Heavy system stats (CPU, memory, disk, GPU) come from the
        periodically refreshed cache (_heavy_stats_cache).
        """
        # active projects in the orchestrator
        active_projects = {
            p: [
                {"unique_id": c.unique_id, "user": c.user, "kind": c.kind, "time": c.time}
                for c in self.projects[p].computing
            ]
            for p in self.projects
        }

        # running processes in the queue
        queue = {
            i.unique_id: i.model_dump()
            for i in self.queue.state()
            if i.state in ["pending", "running"]
        }

        # use cached heavy stats
        stats = self._heavy_stats_cache

        return ServerStateModel(
            version=__version__,
            mode=str(config.mode),
            cpu_only=config.cpu_only,
            active_projects=active_projects,
            queue=queue,
            gpu=stats["gpu"],
            cpu=stats["cpu"],
            memory=stats["memory"],
            disk=stats["disk"],
            mail_available=self.messages.mail_available,
            messages=self.messages.get_messages_system(),
        )

    def exists(self, project_name: str, include_toy_datasets: bool = False) -> bool:
        """
        Test if a project exists in the database
        with a sluggified form (to be able to use it in URL)
        """
        return slugify(project_name) in self.existing_projects()

    def check_project_name(self, project_name: str) -> str:
        """
        Check if a project name is valid
        """
        project_slug = slugify(project_name)
        if self.exists(project_slug):
            raise Exception("This project already exists")
        if project_slug == "":
            raise Exception("The project name is not valid - empty")
        return project_slug

    def create_access_token(self, data: dict, expires_min: int = 60) -> str:
        """
        Create access token
        """
        # create the token
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=expires_min)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, config.secret_key, algorithm=self.jwt_algorithm)

        # add it in the database as active
        self.db_manager.projects_service.add_token(encoded_jwt, "active")

        # logins are rare enough to piggyback the tokens-table cleanup here
        try:
            self.db_manager.projects_service.prune_old_tokens()
        except Exception as e:
            print(f"Error while pruning old tokens: {e}")

        return encoded_jwt

    def revoke_access_token(self, token) -> None:
        """
        Revoke existing access token
        """
        self.db_manager.projects_service.revoke_token(token)

    def decode_access_token(self, token: str) -> dict:
        """
        Decode access token
        """
        # get status
        try:
            status = self.db_manager.projects_service.get_token_status(token)
        except DBException as e:
            raise Exception from e

        if status != "active":
            raise Exception("Token is invalid")

        # decode payload
        payload = jwt.decode(token, config.secret_key, algorithms=[self.jwt_algorithm])
        return payload

    def manage_fifo_queue(self) -> None:
        """
        Manage the current projects in memory for an orchestrator
        """
        if len(self.projects) >= self.max_projects:
            # only idle projects are evictable
            idle_projects = sorted(
                [
                    (p, self.projects[p].starting_time)
                    for p in self.projects
                    if not self.projects[p].computing
                ],
                key=lambda x: x[1],
            )
            if (
                idle_projects and idle_projects[0][1] < time.time() - 600
            ):  # check if the project has a least ten minutes old to avoid destroying current projects
                del self.projects[idle_projects[0][0]]
                print(f"Delete project {idle_projects[0][0]} to gain memory")
            else:
                print("Too many projects in the current memory")
                raise Exception(
                    "There is too many projects currently loaded in this server. Please wait"
                )

    def start_project(self, project_slug: str) -> None:
        """
        Load project in server
        """
        if not self.exists(project_slug):
            raise Exception("This project does not exist")

        try:
            self.projects[project_slug] = Project(
                project_slug,
                self.queue,
                self.db_manager,
                path_models=self.path_models,
                users=self.users,
                messages=self.messages,
            )
        except Exception as e:
            raise Exception(
                f"Error while loading project {project_slug}: {e} - {traceback.format_exc()}"
            ) from e

    def stop_project(self, project_slug: str) -> None:
        """
        Stop a project
        """
        if project_slug not in self.projects:
            return None
        del self.projects[project_slug]

    def stop_process(self, process_id: str, username: str) -> None:
        """
        Stop a specific process
        """
        self.queue.kill(process_id)
        self.log_action(username, f"KILL PROCESS: {process_id}", "all")

    def stop_user_processes(
        self, username: str, project_slug: str | None = None, kind: str | list[str] | None = None
    ) -> None:
        """
        Stop all the processes of a user
        """

        # define the processes to kill
        if kind == "all":
            kind = [
                "train_bert",
                "predict_bert",
                "train_image",
                "predict_image",
                "generation",
                "feature",
                "bertopic",
                "train_quickmodel",
                "predict_quickmodel",
                "predict_with_features",
            ]
        if kind == "bert":
            kind = ["train_bert", "predict_bert"]
        if kind == "image":
            kind = ["train_image", "predict_image"]
        if kind in ("quick", "quickmodel"):
            kind = ["train_quickmodel", "predict_quickmodel", "predict_with_features"]
        if kind is None:
            kind = "all"
        if isinstance(kind, str) and kind != "all":
            kind = [kind]

        # kill all the processes of the user
        if project_slug is None:
            processes = {p: self.projects[p].get_process(kind, username) for p in self.projects}
            for project in processes:
                for process in processes[project]:
                    self.queue.kill(process.unique_id)
                    if process.kind in ("train_bert", "train_image"):
                        process = cast(LMComputing, process)
                        self.db_manager.language_models_service.delete_model(
                            project, process.model_name
                        )
        # kill all the processes of the user for a specific project
        else:
            if project_slug not in self.projects:
                raise Exception("This project is not loaded in memory")
            processes_project = self.projects[project_slug].get_process(kind, username)
            for process in processes_project:
                self.queue.kill(process.unique_id)
                if process.kind in ("train_bert", "train_image"):
                    process = cast(LMComputing, process)
                    self.db_manager.language_models_service.delete_model(
                        project_slug, process.model_name
                    )

    def existing_projects(self) -> list:
        """
        Get existing projects
        """
        return self.db_manager.projects_service.existing_projects()

    def delete_project(self, project_slug: str) -> None:
        """
        Delete a project
        """

        # test if the project exists
        if not self.exists(project_slug):
            raise Exception("This project does not exist")

        # load the project
        if project_slug in self.projects:
            project = self.projects[project_slug]
        else:
            self.start_project(project_slug)
            project = self.projects[project_slug]

        # delete the project
        try:
            project.delete()
        except Exception as e:
            raise Exception from e

        # clean current memory
        if project_slug in self.projects:
            del self.projects[project_slug]

    def duplicate_project(self, source_slug: str, username: str) -> str:
        """
        Kick off the duplication of an existing project. Non-blocking.

        Queues a DuplicateProject task that copies the source project directory
        (data, features, trained model files…) and any static export files.
        The orchestrator update loop picks up the completed copy and clones the
        DB rows in the main process (see `_sync_finish_duplications`). Returns
        the target slug immediately so callers can poll `/projects/status`.
        """
        if not self.exists(source_slug):
            raise Exception("Source project does not exist")

        source_record = self.db_manager.projects_service.get_project(source_slug)
        if source_record is None:
            raise Exception("Source project not found in database")
        source_params: dict[str, Any] = dict(source_record["parameters"])
        source_dir = source_params.get("dir")
        if not source_dir:
            raise Exception("Source project has no directory")
        source_dir_str = str(source_dir)
        if not Path(source_dir_str).exists():
            raise Exception(f"Source project directory missing on disk: {source_dir_str}")

        # find an available target slug, accounting for in-flight duplications
        existing = set(self.existing_projects())
        existing.update(self.project_creation_ongoing.keys())
        existing.update(self.project_duplication_ongoing.keys())
        base = f"{source_slug}-copy"
        target_slug = base
        n = 2
        while target_slug in existing or self.path.joinpath(target_slug).exists():
            target_slug = f"{base}-{n}"
            n += 1

        target_dir = self.path.joinpath(target_slug)
        target_dir_str = str(target_dir)

        # build target parameters: keep everything from source but retarget
        # slug/name/dir so the new project loads against the new directory
        target_params = dict(source_params)
        target_params["project_slug"] = target_slug
        source_name = source_params.get("project_name") or source_slug
        target_params["project_name"] = f"{source_name} (copy)"
        target_params["dir"] = target_dir_str

        source_static = f"{config.data_path}/projects/static/{source_slug}"
        target_static = f"{config.data_path}/projects/static/{target_slug}"
        has_static = Path(source_static).exists()

        task = DuplicateProject(
            source_dir=source_dir_str,
            target_dir=target_dir_str,
            source_static_dir=source_static if has_static else None,
            target_static_dir=target_static if has_static else None,
        )
        task_id = self.queue.add_task("duplicate_project", target_slug, task, queue="cpu")

        self.project_duplication_ongoing[target_slug] = {
            "task_id": task_id,
            "source_slug": source_slug,
            "username": username,
            "source_dir": source_dir_str,
            "target_dir": target_dir_str,
            "target_static": target_static,
            "target_params": target_params,
            "starting_time": time.time(),
        }

        return target_slug

    def _sync_finish_duplications(self) -> None:
        """
        Pick up completed DuplicateProject tasks and finish them by cloning
        the DB rows in the main process.
        """
        to_del: list[str] = []
        for target_slug, info in list(self.project_duplication_ongoing.items()):
            task = self.queue.get(info["task_id"])
            if task is None:
                shutil.rmtree(info["target_dir"], ignore_errors=True)
                shutil.rmtree(info["target_static"], ignore_errors=True)
                self.creation_errors[target_slug] = "Duplication task disappeared"
                to_del.append(target_slug)
                continue

            if task.state in ("pending", "running"):
                continue

            future = task.future
            exception = future.exception() if future is not None and future.done() else None

            if task.state == "cancelled" or exception is not None:
                shutil.rmtree(info["target_dir"], ignore_errors=True)
                shutil.rmtree(info["target_static"], ignore_errors=True)
                self.creation_errors[target_slug] = (
                    f"Duplication failed: {exception}" if exception else "Duplication cancelled"
                )
                self.queue.delete(info["task_id"])
                to_del.append(target_slug)
                continue

            # filesystem copy succeeded — clone the DB rows
            try:
                self.db_manager.projects_service.duplicate_project(
                    source_slug=info["source_slug"],
                    target_slug=target_slug,
                    target_parameters=info["target_params"],
                    user_name=info["username"],
                    source_dir=info["source_dir"],
                    target_dir=info["target_dir"],
                )
                self.log_action(
                    info["username"],
                    f"DUPLICATE PROJECT: {info['source_slug']} -> {target_slug}",
                    target_slug,
                )
            except Exception as e:
                shutil.rmtree(info["target_dir"], ignore_errors=True)
                shutil.rmtree(info["target_static"], ignore_errors=True)
                self.creation_errors[target_slug] = f"DB duplication failed: {e}"

            self.queue.delete(info["task_id"])
            to_del.append(target_slug)

        for slug in to_del:
            self.project_duplication_ongoing.pop(slug, None)

    def clean_unfinished_project(
        self, project_slug: str | None = None, project_name: str | None = None
    ) -> None:
        """
        Clean a project that is not loadable
        Careful : this will delete all the data of the project
        """

        if project_slug is not None:
            project_slug_verif = project_slug
        elif project_name is not None:
            project_slug_verif = slugify(project_name)
        else:
            raise Exception("No project specified")

        # remove from database
        self.db_manager.projects_service.delete_project(project_slug_verif)

        # remove from the server
        if project_slug_verif and project_slug_verif != "":
            shutil.rmtree(self.path.joinpath(project_slug_verif), ignore_errors=True)

            ## remove static files
            if Path(f"{config.data_path}/projects/static/{project_slug_verif}").exists():
                shutil.rmtree(f"{config.data_path}/projects/static/{project_slug_verif}")

        # remove from memory
        if project_slug_verif in self.project_creation_ongoing:
            del self.project_creation_ongoing[project_slug_verif]

    def create_demo_project(self, project_name: str, username: str) -> None:
        """
        Create a demo project for a specific user
        """
        # TODO : put those elements in the config file
        path_data = Path("../frontend/public/gwsd_train_test.csv")
        col_id = "guid"
        col_text = "sentence"
        col_label = "label"

        if not path_data.exists():
            raise Exception("The demo dataset is not available")

        # create name of the project and the directory
        project_name = project_name
        project_slug = slugify(project_name)
        project_path = self.path.joinpath(project_slug)
        os.makedirs(project_path, exist_ok=True)

        # create and save a toy dataset
        df = pd.read_csv(path_data)
        df.to_csv(project_path.joinpath("dataset.csv"), index=False)

        # parameters of the project and creation
        project = ProjectBaseModel(
            project_name=project_name,
            language="en",
            filename="dataset.csv",
            col_id=col_id,
            cols_text=[col_text],
            cols_label=[col_label],
            n_train=1000,
            n_test=500,
        )
        self.starting_project_creation(project, username)

    def available_storage(self, username: str) -> bool:
        """
        Check if the user storage is not exceeded
        """
        limit = self.users.get_storage_limit(username)
        if self.users.get_storage(username) > limit:
            return False
        return True

    def get_toy_datasets(self) -> list[DatasetModel]:
        """
        Get the name of available toy datasets
        """
        toy_datasets = []
        for file in os.listdir(self.path_toy_datasets):
            if file.endswith(".parquet"):
                toy_dataset_name = file.removesuffix(".parquet")
                pq_file = pq.ParquetFile(self.path_toy_datasets.joinpath(file))
                toy_datasets += [
                    DatasetModel(
                        project_slug=toy_dataset_name,
                        columns=[col.name for col in list(pq_file.schema)],
                        n_rows=pq_file.metadata.num_rows,
                    )
                ]
        return toy_datasets


orchestrator = None


def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        orchestrator = Orchestrator()
    return orchestrator
