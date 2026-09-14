import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from threading import Lock

import yaml

from activetigger.config import config
from activetigger.datamodels import (
    AuthUserModel,
    DatasetModel,
    NewUserModel,
    ProjectModel,
    ProjectSummaryModel,
    UserActivityPointModel,
    UserCredentialInput,
    UserCredentialPublic,
    UserInDBModel,
    UserModel,
    UsersStateModel,
    UserStatistics,
)
from activetigger.db.manager import DatabaseManager
from activetigger.errors import AlreadyExistsError, InvalidInputError, NotFoundError
from activetigger.functions import compare_to_hash, decrypt, encrypt, get_dir_size, get_hash
from activetigger.messages import Messages

USERS_PARAMETERS_FILE = "users_parameters.yaml"
DEFAULT_USERS_PARAMETERS = {"root": {"limit": 100}}

# Recursive disk scans for project sizes dominate the cost of the admin
# "all projects" listing; cache them with a short TTL so refreshes are cheap.
_DIR_SIZE_TTL_SECONDS = 300
_DIR_SIZE_CACHE: dict[str, tuple[float, float]] = {}
_DIR_SIZE_CACHE_LOCK = Lock()


def _cached_dir_size(slug: str, path: str) -> float | None:
    now = time.monotonic()
    with _DIR_SIZE_CACHE_LOCK:
        cached = _DIR_SIZE_CACHE.get(slug)
        if cached is not None and now - cached[0] < _DIR_SIZE_TTL_SECONDS:
            return cached[1]
    try:
        size = round(get_dir_size(path), 1)
    except Exception as e:
        print(e)
        return None
    with _DIR_SIZE_CACHE_LOCK:
        _DIR_SIZE_CACHE[slug] = (now, size)
    return size


class Users:
    """
    Managers users
    """

    # parameters for per-user statistics
    ANNOTATION_GAP_CAP_SECONDS = 600  # gaps above this are session breaks
    ANNOTATION_GAP_MIN_SECONDS = 1.0  # gaps below this are batch/import writes
    ANNOTATION_TIMES_LIMIT = 5000  # bound the timestamps query on large tables
    ACTIVITY_DAYS = 7
    GPU_TIME_MIN_GB = 2.0  # device-wide readings below this are driver/baseline noise

    db_manager: DatabaseManager
    users_parameters: dict
    failed_attemps: dict[str, list[datetime]]
    messages: Messages

    def __init__(
        self,
        db_manager: DatabaseManager,
        messages: Messages,
    ):
        """
        Init users references.

        Per-user parameters are loaded once from ``users_parameters.yaml``
        located in ``config.data_path``. The file maps a username to a dict
        of parameters; the ``limit`` key is the user's storage quota in GB.
        If the file does not exist, it is created with a default ``root``
        entry (100 GB).
        """
        self.db_manager = db_manager
        self.messages = messages
        self.users_parameters = self._load_users_parameters()
        self.failed_attemps: dict = {}

    @staticmethod
    def _load_users_parameters() -> dict:
        path = Path(config.data_path) / USERS_PARAMETERS_FILE
        if not path.exists():
            with open(path, "w") as f:
                yaml.safe_dump(DEFAULT_USERS_PARAMETERS, f)
            return dict(DEFAULT_USERS_PARAMETERS)
        with open(path) as f:
            content = yaml.safe_load(f) or {}
        if not isinstance(content, dict):
            return {}
        return content

    def log_failed_login_attempt(self, username: str) -> None:
        """
        Register the timestamp of a failed login attempt
        Raise an error if there are too many failed attempts in a short period
        """
        if username not in self.failed_attemps:
            self.failed_attemps[username] = []
        self.failed_attemps[username].append(datetime.now(timezone.utc))

    def check_failed_login_attempts(
        self, username: str, timewindow: int = 10, max_attempts: int = 3
    ) -> None:
        """
        Check if there are too many failed login attempts in a short period
        Raise an error if there are too many failed attempts in a short period
        """
        if username not in self.failed_attemps:
            return
        # filter attempts in the timewindow
        now = datetime.now(timezone.utc)
        self.failed_attemps[username] = [
            t for t in self.failed_attemps[username] if (now - t).total_seconds() < timewindow
        ]
        if len(self.failed_attemps[username]) >= max_attempts:
            raise Exception("Too many failed login attempts. Please try again after a few minutes.")

    def get_project_auth(self, project_slug: str) -> dict[str, str]:
        """
        Get user auth for a project
        """
        return self.db_manager.projects_service.get_project_auth(project_slug)

    def set_auth(self, auth: AuthUserModel) -> None:
        """
        Set user auth for a project
        """
        if auth.status is None:
            raise InvalidInputError("Missing status")
        self.get_user(auth.username)
        if self.db_manager.projects_service.get_project(auth.project_slug) is None:
            raise NotFoundError("Project not found")
        self.db_manager.projects_service.add_auth(auth.project_slug, auth.username, auth.status)

    def delete_auth(self, username: str, project_slug: str) -> None:
        """
        Delete user auth
        """
        self.get_user(username)
        self.db_manager.projects_service.delete_auth(project_slug, username)

    def get_auth_projects(self, username: str, auth: str | None = None) -> list:
        """
        Get user auth
        """
        return self.db_manager.projects_service.get_user_auth_projects(username, auth)

    def get_auth(self, username: str, project_slug: str = "all") -> list:
        """
        Get user auth
        Comments:
        - Either for all projects
        - Or one project
        """
        if project_slug == "all":
            auth = self.db_manager.projects_service.get_user_auth(username)
        else:
            auth = self.db_manager.projects_service.get_user_auth(username, project_slug)
        return auth

    def existing_users(self, username: str = "root", active: bool = True) -> dict[str, UserModel]:
        """
        Get existing users which have been created by one user
        (except root which can't be modified)
        TODO : better rules
        """
        if username == "root":
            return self.db_manager.users_service.get_users_created_by("all", active)
        else:
            # TODO : in the future, allows users in project related to the account ?
            return {}

    def add_user(
        self,
        new_user: NewUserModel,
        created_by: str,
    ) -> None:
        """
        Add user to database
        Comments:
            Default, users are managers
        """
        # test if the user doesn't exist, even among deactivated users
        if new_user.username in self.existing_users(active=False):
            raise AlreadyExistsError("Username already exists")
        hash_pwd = get_hash(new_user.password)
        self.db_manager.users_service.add_user(
            new_user.username,
            hash_pwd.decode("utf8"),
            new_user.status,
            created_by,
            contact=new_user.contact,
        )

    def delete_user(self, user_to_delete: str, username: str) -> None:
        """
        Deleting user
        """
        # test specific rights
        if user_to_delete == "root":
            raise Exception("Can't delete root user")
        if user_to_delete not in self.existing_users():
            raise NotFoundError("Username does not exist")
        if user_to_delete not in self.existing_users(username):
            raise Exception("You don't have the right to delete this user")

        # delete the user
        self.db_manager.users_service.delete_user(user_to_delete)

    def get_user(self, name: str) -> UserInDBModel:
        """
        Get active user from database
        """
        if name not in self.existing_users():
            raise NotFoundError("Username doesn't exist or is deactivated")
        user = self.db_manager.users_service.get_user(name)
        return UserInDBModel(
            username=name,
            hashed_password=user.key,
            status=(user.informations or {}).get("status"),
        )

    def authenticate_user(self, username: str, password: str) -> UserInDBModel:
        """
        User authentification
        - Check too many failed login attempts
        - Check username/password
        """
        self.check_failed_login_attempts(username)
        try:
            user = self.get_user(username)
            if not compare_to_hash(password, user.hashed_password):
                raise InvalidInputError("Wrong password")
            return user
        except Exception:
            self.log_failed_login_attempt(username)
            raise InvalidInputError("Wrong username or password")

    def auth(self, username: str, project_slug: str) -> str | None:
        """
        Check auth for a specific project
        """
        user_auth = self.get_auth(username, project_slug)
        if len(user_auth) == 0:  # not associated
            return None
        return user_auth[0][1]

    def change_password(
        self, username: str, password_old: str, password1: str, password2: str
    ) -> None:
        """
        Change password for a user
        """
        if password1 != password2:
            raise Exception("Passwords don't match")
        user = self.get_user(username)
        if not compare_to_hash(password_old, user.hashed_password):
            raise InvalidInputError("Wrong password")
        hash_pwd = get_hash(password1)
        self.db_manager.users_service.change_password(username, hash_pwd.decode("utf8"))
        return None

    def change_email(self, username: str, new_email: str, password: str) -> None:
        """
        Change contact email for a user.
        Requires the current password to confirm the action.
        """
        new_email = new_email.strip()
        if not new_email or "@" not in new_email:
            raise InvalidInputError("Invalid email address")
        user = self.get_user(username)
        if not compare_to_hash(password, user.hashed_password):
            raise InvalidInputError("Wrong password")
        try:
            existing = self.db_manager.users_service.get_user_by_mail(new_email)
        except Exception:
            existing = None
        if existing is not None and existing != username:
            raise Exception("Email already used by another account")
        self.db_manager.users_service.change_contact(username, new_email)

    def get_contact(self, username: str) -> str:
        """
        Get contact email for a user (empty string if not set)
        """
        user = self.db_manager.users_service.get_user(username)
        return user.contact or ""

    def list_credentials(self, username: str) -> list[UserCredentialPublic]:
        """
        List saved endpoint/credentials entries, without the secrets
        """
        informations = self.db_manager.users_service.get_informations(username)
        return [
            UserCredentialPublic(
                name=name, api=entry.get("api", ""), endpoint=entry.get("endpoint")
            )
            for name, entry in informations.get("credentials", {}).items()
        ]

    def add_credentials(self, username: str, credential: UserCredentialInput) -> None:
        """
        Save an endpoint/credentials entry in the user informations.
        The secret is encrypted and never sent back to the client.
        An existing entry with the same name is replaced.
        """
        if credential.name.strip() == "":
            raise Exception("You should provide a name")
        informations = self.db_manager.users_service.get_informations(username)
        credentials = dict(informations.get("credentials", {}))
        credentials[credential.name.strip()] = {
            "api": credential.api,
            "endpoint": credential.endpoint,
            "credentials": encrypt(credential.credentials, config.secret_key),
        }
        informations["credentials"] = credentials
        self.db_manager.users_service.update_informations(username, informations)

    def delete_credentials(self, username: str, name: str) -> None:
        """
        Delete a saved endpoint/credentials entry
        """
        informations = self.db_manager.users_service.get_informations(username)
        credentials = dict(informations.get("credentials", {}))
        if name not in credentials:
            raise NotFoundError(f"Credentials {name} not found")
        del credentials[name]
        informations["credentials"] = credentials
        self.db_manager.users_service.update_informations(username, informations)

    def resolve_credentials(self, username: str, name: str) -> tuple[str | None, str]:
        """
        Get the (endpoint, decrypted secret) of a saved entry.
        Backend use only: never expose the result in a route.
        """
        informations = self.db_manager.users_service.get_informations(username)
        entry = informations.get("credentials", {}).get(name)
        if entry is None:
            raise NotFoundError(f"Credentials {name} not found")
        return entry.get("endpoint"), decrypt(entry["credentials"], config.secret_key)

    def force_change_password(self, username: str, password: str) -> None:
        """
        Force change password for a user (no old password needed)
        """
        hash_pwd = get_hash(password)
        self.db_manager.users_service.change_password(username, hash_pwd.decode("utf8"))

    def admin_reset_password(self, target_username: str) -> str:
        """
        Generate a random password for a user, persist it and return it
        (in plain text) so the caller can hand it back to the user once.
        """
        if target_username == "root":
            raise Exception("Cannot reset root password from here")
        # Ensures the user exists and is active
        self.get_user(target_username)
        new_password = secrets.token_urlsafe(12)
        self.force_change_password(target_username, new_password)
        return new_password

    def get_statistics(self, username: str) -> UserStatistics:
        """
        Get statistics for specific user: authorized projects, total
        annotations, recent hourly activity, median annotation time and
        GPU/compute time of completed processes.
        """
        try:
            projects = {i[0]: i[1] for i in self.get_auth_projects(username)}
            total_annotations = self.db_manager.users_service.count_annotations(username)
            times = self.db_manager.users_service.get_annotation_times(
                username, limit=self.ANNOTATION_TIMES_LIMIT
            )
            gpu_time, compute_time = self._compute_process_times(username)
            return UserStatistics(
                username=username,
                projects=projects,
                total_annotations=total_annotations,
                gpu_time_seconds=gpu_time,
                compute_time_seconds=compute_time,
                median_annotation_time_seconds=self._median_annotation_gap(times),
                annotation_activity=self._build_hourly_activity(username, self.ACTIVITY_DAYS),
            )
        except Exception as e:
            raise Exception(f"Error in getting statistics for {username}") from e

    def _median_annotation_gap(self, times: list[datetime]) -> float | None:
        """
        Median gap in seconds between consecutive annotations, ignoring
        session breaks (gaps above ANNOTATION_GAP_CAP_SECONDS) and
        batch/import writes (gaps below ANNOTATION_GAP_MIN_SECONDS).
        None if not enough data.
        """
        if len(times) < 2:
            return None
        ordered = sorted(times)
        gaps = [
            gap
            for previous, current in zip(ordered, ordered[1:])
            if self.ANNOTATION_GAP_MIN_SECONDS
            <= (gap := (current - previous).total_seconds())
            <= self.ANNOTATION_GAP_CAP_SECONDS
        ]
        if len(gaps) < 2:
            return None
        return float(median(gaps))

    def _compute_process_times(self, username: str) -> tuple[float, float]:
        """
        (gpu_time_seconds, compute_time_seconds) over the user's completed
        processes. GPU time sums the durations of processes whose events
        report a peak GPU memory above GPU_TIME_MIN_GB (the reading is
        device-wide, so lower values are baseline noise); compute time
        sums all durations.
        """
        processes = self.db_manager.monitoring_service.get_completed_processes(
            kind="all", username=username, limit=1000
        )
        gpu_time = 0.0
        compute_time = 0.0
        for process in processes:
            if process.duration is None:
                continue
            compute_time += float(process.duration)
            events = process.events if isinstance(process.events, dict) else {}
            gpu_event = events.get("gpu")
            try:
                max_used_gb = (
                    float(gpu_event.get("max_used_gb") or 0.0)
                    if isinstance(gpu_event, dict)
                    else 0.0
                )
            except (TypeError, ValueError):
                max_used_gb = 0.0
            if max_used_gb > self.GPU_TIME_MIN_GB:
                gpu_time += float(process.duration)
        return gpu_time, compute_time

    def _build_hourly_activity(self, username: str, days: int) -> list[UserActivityPointModel]:
        """
        Zero-filled hourly annotation counts for the user over the last `days`
        days (same series construction as Monitoring._compute_weekly_activity).
        """
        annotations_by_hour, _ = self.db_manager.monitoring_service.get_hourly_activity_counts(
            days=days, user_name=username
        )
        total_hours = days * 24
        now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_hour = now_hour - timedelta(hours=total_hours - 1)
        return [
            UserActivityPointModel(
                hour=(start_hour + timedelta(hours=i)).isoformat(),
                annotations=annotations_by_hour.get(start_hour + timedelta(hours=i), 0),
            )
            for i in range(total_hours)
        ]

    def get_storage(self, username: str) -> float:
        """
        Get total size for user projects in GB
        """
        projects = self.db_manager.users_service.get_user_created_projects(username)
        size_mb = sum(
            [get_dir_size(f"{config.data_path}/projects/{project}") for project in projects]
        )
        return size_mb / 1024

    def get_storage_limit(self, username: str) -> float:
        """
        Get storage limit (GB) for user from `users_parameters.yaml`,
        falling back to `config.user_hdd_max` if the user has no entry.
        """
        params = self.users_parameters.get(username)
        if params and "limit" in params:
            return float(params["limit"])
        return config.user_hdd_max

    def state(self, project_slug: str) -> UsersStateModel:
        """
        Get last annotation date for all users
        """
        r = self.db_manager.users_service.get_project_users_last_annotation(project_slug)
        return UsersStateModel(
            users=list(r.keys()),
            last_schemes={i: r[i].scheme for i in r},
        )

    def get_auth_datasets(self, username: str) -> list[DatasetModel]:
        """
        Get datasets authorized for the user (projects where is manager)
        """
        projects = self.get_auth_projects(username, auth="manager")
        return [
            DatasetModel(
                project_slug=p[2]["project_slug"],
                columns=p[2]["all_columns"],
                n_rows=p[2]["n_total"],
            )
            for p in projects
        ]

    def get_user_projects(self, username: str) -> list[ProjectSummaryModel]:
        """
        Get projects authorized for the user
        """
        projects_auth = self.get_auth_projects(username)
        projects = []
        for i in list(reversed(projects_auth)):
            # get the project slug
            project_slug = i[0]
            user_right = i[1]
            parameters = self.db_manager.projects_service.get_project(project_slug)
            if parameters is None:
                continue
            parameters = ProjectModel(**parameters["parameters"])
            created_by = i[3]
            created_at = i[4].strftime("%Y-%m-%d %H:%M:%S")
            try:
                size = round(get_dir_size(config.data_path + "/projects/" + i[0]), 1)
            except Exception as e:
                print(e)
                size = None
            last_activity = self.db_manager.logs_service.get_last_activity_project(i[0])

            projects.append(
                ProjectSummaryModel(
                    project_slug=project_slug,
                    user_right=user_right,
                    parameters=parameters,
                    created_by=created_by,
                    created_at=created_at,
                    size=size,
                    last_activity=last_activity,
                )
            )
        return projects

    def get_all_projects(self, username: str) -> list[ProjectSummaryModel]:
        """
        Get all existing projects regardless of auth (admin view).
        user_right reflects the given user's auth on each project, or "none".
        """
        rows = self.db_manager.projects_service.existing_projects_with_meta()
        last_activity_map = self.db_manager.logs_service.get_last_activity_all_projects()
        auths_map = self.db_manager.projects_service.get_user_auths_all_projects(username)

        slugs = [r["project_slug"] for r in rows]
        projects_root = config.data_path + "/projects/"
        # Disk scans are I/O bound; threads parallelize them well even under the GIL.
        with ThreadPoolExecutor(max_workers=min(16, max(4, len(slugs)))) as pool:
            sizes = list(pool.map(lambda s: _cached_dir_size(s, projects_root + s), slugs))

        projects = []
        for row, size in zip(rows, sizes):
            slug = row["project_slug"]
            projects.append(
                ProjectSummaryModel(
                    project_slug=slug,
                    user_right=auths_map.get(slug, "none"),
                    parameters=ProjectModel(**row["parameters"]),
                    created_by=row["user_name"],
                    created_at=row["time_created"].strftime("%Y-%m-%d %H:%M:%S"),
                    size=size,
                    last_activity=last_activity_map.get(slug),
                )
            )
        projects.sort(key=lambda p: p.created_at, reverse=True)
        return projects

    def reset_password(self, mail: str) -> None:
        """
        Reset password for a user with the given email
        """
        # Check if mail is connected to a user
        user_name = self.db_manager.users_service.get_user_by_mail(mail)

        # Generate a random password
        new_password = secrets.token_hex(16)

        # Send the mail to the user with the new password
        self.messages.send_mail_reset_password(user_name, mail, new_password)

        # Update the user's password in the database
        self.force_change_password(user_name, new_password)
