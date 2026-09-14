from typing import Annotated

from fastapi import APIRouter, Depends, Query

from activetigger.app.dependencies import ServerAction, test_rights, verified_user
from activetigger.datamodels import (
    MonitoringActivityModel,
    MonitoringMetricsModel,
    ProjectSummaryModel,
    UserInDBModel,
)
from activetigger.orchestrator import get_orchestrator

router = APIRouter(tags=["monitoring"])


@router.get("/monitoring/metrics")
def get_monitoring_metrics(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> MonitoringMetricsModel:
    """
    Get monitoring metrics
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    metrics = get_orchestrator().monitoring.get_metrics()
    return metrics


@router.get("/monitoring/data")
def get_monitoring_data(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], kind: str
) -> list:
    """
    Get monitoring data
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    data = get_orchestrator().monitoring.get_data(kind)
    return data.to_dict(orient="records")


@router.get("/monitoring/projects")
def get_all_projects(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list[ProjectSummaryModel]:
    """
    Get summary of all existing projects (admin view).
    user_right reflects current user's auth on each project, or "none".
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    return get_orchestrator().users.get_all_projects(current_user.username)


@router.get("/monitoring/activity")
def get_monitoring_activity(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    days: int = Query(7, ge=1, le=3650),
) -> MonitoringActivityModel:
    """
    Hourly timeline of the instance activity: annotations made and
    distinct users acting per hour over the last `days` days.
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    return get_orchestrator().monitoring.get_weekly_activity(days=days)
