from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from activetigger.app.dependencies import ProjectAction, get_project, test_rights, verified_user
from activetigger.datamodels import (
    BertopicProjectionData,
    BertopicTopicsOutModel,
    ComputeBertopicModel,
    UserInDBModel,
)
from activetigger.errors import APIError
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project

router = APIRouter(tags=["BERTopic"])


@router.post("/bertopic/compute", dependencies=[Depends(verified_user)])
def compute_bertopic(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bertopic: ComputeBertopicModel,
) -> str:
    """
    Compute BERTopic model for the project.
    """
    if not project.bertopic.name_available(bertopic.name):
        raise HTTPException(
            status_code=400,
            detail=f"BERTopic model with name '{bertopic.name}' already exists (after slugification).",
        )
    if project.params.dir is None:
        raise HTTPException(
            status_code=400,
            detail="Project dataset path is not set. Cannot compute BERTopic model.",
        )

    # Force the language of the project
    bertopic.language = project.params.language

    if project.params.kind == "image":
        raise HTTPException(status_code=400, detail="BERTopic is not supported for image projects")
    try:
        unique_id = project.bertopic.compute(
            path_data=project.params.dir,
            col_id=None,
            col_text="text",
            parameters=bertopic,
            name=bertopic.name,
            user=current_user.username,
        )
        get_orchestrator().log_action(
            current_user.username, f"COMPUTE BERTopic MODEL: {bertopic.name}", project.name
        )
        project.monitoring.register_process(
            process_name=unique_id,
            kind="fit_bertopic",
            parameters={},
            user_name=current_user.username,
        )
        return unique_id
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bertopic/topics", dependencies=[Depends(verified_user)])
def get_bertopic_topics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> BertopicTopicsOutModel:
    """
    Get topics from the BERTopic model for the project.
    """
    if project.params.kind == "image":
        raise HTTPException(status_code=400, detail="BERTopic is not supported for image projects")
    try:
        return BertopicTopicsOutModel(
            topics=project.bertopic.get_topics(name=name),
            parameters=project.bertopic.get_parameters(name=name),
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bertopic/projection", dependencies=[Depends(verified_user)])
def get_bertopic_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> BertopicProjectionData:
    """
    Get projection from the BERTopic model for the project.
    """
    if project.params.kind == "image":
        raise HTTPException(status_code=400, detail="BERTopic is not supported for image projects")
    try:
        return project.bertopic.get_projection(name=name)
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bertopic/delete", dependencies=[Depends(verified_user)])
def delete_bertopic_model(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> None:
    """
    Delete a BERTopic model for the project.
    """
    if project.params.kind == "image":
        raise HTTPException(status_code=400, detail="BERTopic is not supported for image projects")
    try:
        project.bertopic.delete(name=name)
        get_orchestrator().log_action(
            current_user.username, f"DELETE BERTopic MODEL: {name}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bertopic/export-to-scheme", dependencies=[Depends(verified_user)])
def export_bertopic_to_scheme(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    topic_model_name: str = Query(...),
) -> None:
    """
    Export the topic model as a scheme for the train set
    """
    if project.params.kind == "image":
        raise HTTPException(status_code=400, detail="BERTopic is not supported for image projects")
    orchestrator = get_orchestrator()
    try:
        test_rights(ProjectAction.ADD, current_user.username, project.name)

        labels, clusters, topic_id_to_topic_name = project.bertopic.export_to_scheme(
            topic_model_name
        )

        new_scheme_name = f"topic-model-{topic_model_name}"

        # add a new scheme
        project.schemes.add_scheme(
            name=new_scheme_name,
            labels=labels,
            user=current_user.username,
        )

        # Transform the annotation into the right format
        elements = [
            {"element_id": el_id, "annotation": topic_id_to_topic_name[cluster], "comment": ""}
            for (el_id, cluster) in clusters.items()
            if cluster != -1
        ]
        project.schemes.projects_service.add_annotations(
            dataset="train",
            user_name=current_user.username,
            project_slug=project.name,
            scheme=new_scheme_name,
            elements=elements,
        )

        orchestrator.log_action(
            current_user.username, f"Export BERTopic to scheme : {new_scheme_name}", project.name
        )

    except Exception as e:
        orchestrator.log_action(current_user.username, f"DEBUG-EXPORT-TO-SCHEME: {e}", project.name)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bertopic/export-to-feature", dependencies=[Depends(verified_user)])
def export_bertopic_to_feature(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    topic_model_name: str = Query(...),
) -> None:
    """
    Export the topic model as a feature for quick models
    """
    if project.params.kind == "image":
        raise HTTPException(status_code=400, detail="BERTopic is not supported for image projects")
    orchestrator = get_orchestrator()
    try:
        test_rights(ProjectAction.ADD, current_user.username, project.name)

        # get topics and clusters
        labels, clusters, topic_id_to_topic_name = project.bertopic.export_to_scheme(
            topic_model_name
        )

        feature_name = f"bertopic_{topic_model_name}"

        # if the feature already exists, delete it first
        if project.features.exists(feature_name):
            project.features.delete(feature_name)

        # build a Series aligned to the full features index
        features_index = pd.read_parquet(project.features.path_features, columns=[]).index

        # map each id to its topic name, "unassigned" for outliers or missing
        topic_series = pd.Series("unassigned", index=features_index, dtype=str)
        for id_internal, cluster_id in clusters.items():
            str_id = str(id_internal)
            if str_id in topic_series.index and cluster_id != -1:
                topic_series[str_id] = topic_id_to_topic_name[cluster_id]

        # one-hot encode
        dummies = pd.get_dummies(topic_series, drop_first=True).astype(int)

        # sanitize column names (__ is the feature separator)
        dummies.columns = [c.replace("__", "_") for c in dummies.columns]

        project.features.add(
            name=feature_name,
            kind="bertopic",
            username=current_user.username,
            parameters={"topic_model": topic_model_name},
            new_content=dummies,
        )

        orchestrator.log_action(
            current_user.username,
            f"Export BERTopic to feature : {feature_name}",
            project.name,
        )

    except Exception as e:
        orchestrator.log_action(
            current_user.username, f"DEBUG-EXPORT-TO-FEATURE: {e}", project.name
        )
        raise HTTPException(status_code=500, detail=str(e))
