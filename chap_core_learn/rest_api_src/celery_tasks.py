import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Generic

from celery import Celery, Task, shared_task
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel
from redis import Redis
from sqlalchemy import create_engine

from ..database.database import SessionWrapper
from ..worker.interface import ReturnType

# Get Celery-compatible logger
logger = get_task_logger(__name__)
logger.setLevel(logging.INFO)


# ---------- CONFIGURATION ----------


def read_environment_variables():
    """Load environment variables using dotenv and return Redis URL."""
    load_dotenv(find_dotenv())
    return os.getenv("CELERY_BROKER", "redis://localhost:6379")


# Setup Celery application
url = read_environment_variables()
logger.info(f"Connecting to {url}")
app = Celery("worker", broker=url, backend=url)

# Celery config: use pickle for complex Python objects
app.conf.update(
    task_serializer="pickle",
    accept_content=["pickle"],
    result_serializer="pickle",
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
)

# Connect to Redis for job metadata tracking (separate DB 2)
redis_url = "redis" if "localhost" not in url else "localhost"
r = Redis(host=redis_url, port=6379, db=2, decode_responses=True)


# ---------- JOB TRACKING ----------


class JobDescription(BaseModel):
    """Metadata model for tracked jobs."""

    id: str
    type: str
    name: str
    status: str
    start_time: str | None
    end_time: str | None
    result: str | None


# ---------- TASK WRAPPER ----------


class TrackedTask(Task):
    """Celery Task subclass that adds per-task logging and metadata tracking."""

    def __call__(self, *args, **kwargs):
        task_id = self.request.id

        # Setup per-task log file
        log_path = Path("logs") / f"task_{task_id}.txt"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

        # Backup current log handlers
        old_handlers = logger.handlers[:]
        old_root_handlers = logging.getLogger().handlers[:]

        # Apply per-task handlers
        logger.handlers = [file_handler, logging.StreamHandler()]
        logging.getLogger().addHandler(file_handler)

        try:
            r.hmset(f"job_meta:{task_id}", {"status": "STARTED"})
            return super().__call__(*args, **kwargs)
        finally:
            file_handler.close()
            logger.handlers = old_handlers
            logging.getLogger().handlers = old_root_handlers

    def apply_async(self, args=None, kwargs=None, **options):
        job_name = kwargs.pop(JOB_NAME_KW, "Unnamed")
        job_type = kwargs.pop(JOB_TYPE_KW, "Unspecified")

        result = super().apply_async(args=args, kwargs=kwargs, **options)

        r.hmset(
            f"job_meta:{result.id}",
            {"job_name": job_name, "job_type": job_type, "status": "PENDING", "start_time": datetime.now().isoformat()},
        )
        return result

    def on_success(self, retval, task_id, args, kwargs):
        r.hmset(
            f"job_meta:{task_id}",
            {"status": "SUCCESS", "result": json.dumps(retval), "end_time": datetime.now().isoformat()},
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        r.hmset(
            f"job_meta:{task_id}",
            {
                "status": "FAILURE",
                "error": str(exc),
                "traceback": str(einfo.traceback),
                "end_time": datetime.now().isoformat(),
            },
        )


# ---------- SHARED TASKS ----------


@shared_task(name="celery.ping")
def ping():
    """Basic health check."""
    return "pong"


@app.task(base=TrackedTask)
def celery_run(func, *args, **kwargs):
    """Run a Python function asynchronously."""
    return func(*args, **kwargs)


ENGINES_CACHE = {}


@app.task(base=TrackedTask)
def celery_run_with_session(func, *args, **kwargs):
    """Run a Python function that requires a SQLAlchemy session."""
    database_url = kwargs.pop("database_url")
    if database_url not in ENGINES_CACHE:
        ENGINES_CACHE[database_url] = create_engine(database_url)
    with SessionWrapper(ENGINES_CACHE[database_url]) as session:
        return func(*args, **kwargs | {"session": session})


# ---------- JOB TRACKING WRAPPER ----------

JOB_TYPE_KW = "__job_type__"
JOB_NAME_KW = "__job_name__"


class CeleryJob(Generic[ReturnType]):
    """Wrapper for an individual Celery job."""

    def __init__(self, job, app: Celery):
        self._job = job
        self._app = app

    @property
    def _result(self) -> AsyncResult:
        return AsyncResult(self._job.id, app=self._app)

    @property
    def status(self) -> str:
        return self._result.state

    @property
    def result(self) -> ReturnType:
        return self._result.result

    @property
    def id(self) -> str:
        return self._job.id

    @property
    def is_finished(self) -> bool:
        return self._result.state in ("SUCCESS", "FAILURE")

    @property
    def exception_info(self) -> str:
        return str(self._result.traceback or "")

    def cancel(self):
        self._result.revoke()

    def get_logs(self) -> Optional[str]:
        """Return logs from task-specific log file and append traceback if failed."""
        log_path = Path("app/logs") / f"task_{self._job.id}.txt"
        if log_path.exists():
            logs = log_path.read_text()
            job_meta = get_job_meta(self.id)
            if job_meta["status"] == "FAILURE":
                logs += "\n" + job_meta["traceback"]
            return logs
        return None


class CeleryPool(Generic[ReturnType]):
    """Simplified task pool abstraction for launching Celery jobs."""

    def __init__(self, celery: Celery = None):
        assert celery is None  # only supports singleton for now
        self._celery = app

    def queue(self, func: Callable[..., ReturnType], *args, **kwargs) -> CeleryJob[ReturnType]:
        job = celery_run.delay(func, *args, **kwargs)
        return CeleryJob(job, app=self._celery)

    def queue_db(self, func: Callable[..., ReturnType], *args, **kwargs) -> CeleryJob[ReturnType]:
        job = celery_run_with_session.delay(func, *args, **kwargs)
        return CeleryJob(job, app=self._celery)

    def get_job(self, task_id: str) -> CeleryJob[ReturnType]:
        return CeleryJob(AsyncResult(task_id, app=self._celery), app=self._celery)

    def list_jobs(self, status: str = None):
        """Return list of jobs tracked in Redis, optionally filtering by status."""
        keys = r.keys("job_meta:*")
        jobs = []

        for key in keys:
            task_id = key.split(":")[1]
            meta = r.hgetall(key)
            meta["task_id"] = task_id
            if status is None or meta.get("status") == status:
                jobs.append(meta)

        return [
            JobDescription(
                id=meta["task_id"],
                type=meta.get("job_type", "Unspecified"),
                name=meta.get("job_name", "Unnamed"),
                status=meta["status"],
                start_time=meta.get("start_time"),
                end_time=meta.get("end_time"),
                result=meta.get("result"),
            )
            for meta in sorted(jobs, key=lambda x: x.get("start_time", datetime(1900, 1, 1).isoformat()), reverse=True)
        ]


# ---------- UTILITY ----------


def get_job_meta(task_id: str):
    """Fetch job metadata from Redis."""
    key = f"job_meta:{task_id}"
    return r.hgetall(key) if r.exists(key) else None
