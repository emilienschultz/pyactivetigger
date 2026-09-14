from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from activetigger.app.dependencies import (
    ProjectAction,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    MODEL_NAME_PATTERN,
    BertModelModel,
    ImageModelModel,
    ModelInformationsModel,
    NerModelModel,
    QuickModelInModel,
    QuickModelOutModel,
    TextDatasetModel,
    UserInDBModel,
)
from activetigger.errors import APIError, InvalidInputError
from activetigger.orchestrator import get_orchestrator
from activetigger.project import Project
from activetigger.uploads import get_upload_staging

router = APIRouter(tags=["models"])

# Reusable query-param validator for any user-supplied model name. Mirrors
# the same regex used in BertModelModel.name / ImageModelModel.name so query
# routes (delete, rename) can't slip a "../" past the body validator.

ModelName = Annotated[str, Query(pattern=MODEL_NAME_PATTERN)]


@router.post("/models/quick/train", dependencies=[Depends(verified_user)])
def train_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    quickmodel: QuickModelInModel,
) -> None:
    """
    Compute quickmodel
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        project.train_quickmodel(quickmodel, current_user.username)
        get_orchestrator().log_action(
            current_user.username, f"TRAIN SIMPLE MODEL {quickmodel.name}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quick/retrain", dependencies=[Depends(verified_user)])
def retrain_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    name: str,
) -> None:
    """
    Retrain quickmodel
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        project.retrain_quickmodel(name, scheme, current_user.username)
        get_orchestrator().log_action(
            current_user.username, f"RETRAIN SIMPLE MODEL {name}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quick/delete", dependencies=[Depends(verified_user)])
def delete_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> None:
    """
    Delete quickmodel
    """
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    try:
        project.quickmodels.delete(name)
        get_orchestrator().log_action(
            current_user.username, f"DELETE SIMPLE MODEL + FEATURES: {name}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quick/rename", dependencies=[Depends(verified_user)])
def rename_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: ModelName,
    new_name: ModelName,
) -> None:
    """
    Rename quickmodel
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.quickmodels.rename(former_name, new_name)
        get_orchestrator().log_action(
            current_user.username,
            f"INFO RENAME QUICK MODEL: {former_name} -> {new_name}",
            project.name,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/quick", dependencies=[Depends(verified_user)])
def get_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> QuickModelOutModel | None:
    """
    Get available quickmodel by a name
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        sm = project.quickmodels.get(name)
        return QuickModelOutModel(
            name=sm.name,
            model=sm.model_type,
            params=sm.model_params,
            features=sm.features,
            statistics_train=sm.statistics_train,
            statistics_test=sm.statistics_test,
            statistics_cv10=sm.statistics_cv10,
            balance_classes=sm.balance_classes,
            scheme=sm.scheme,
            username=sm.user,
            exclude_labels=sm.exclude_labels if hasattr(sm, "exclude_labels") else [],
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/information", dependencies=[Depends(verified_user)])
def get_model_information(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: ModelName,
    kind: str,
) -> ModelInformationsModel:
    """
    Get model information.

    Guarded by ProjectAction.GET so an authenticated user can't read
    parameters / metrics for a project they don't have access to.
    """
    test_rights(ProjectAction.GET, current_user.username, project.name)
    try:
        if kind == "bert":
            return project.languagemodels.get_informations(name)
        elif kind == "quick":
            return project.quickmodels.get_informations(name)
        elif kind == "image":
            if project.imagemodels is None:
                raise Exception("Image models are only available for image projects")
            return project.imagemodels.get_informations(name)
        elif kind == "ner":
            if project.nermodels is None:
                raise Exception("NER models are not available for this project")
            return project.nermodels.get_informations(name)
        else:
            raise InvalidInputError(f"Model kind {kind} not recognized")
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        print(f"Erreur in /models/information:\n{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/predict", dependencies=[Depends(verified_user)])
def predict(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model_name: ModelName,
    scheme: str,
    kind: str,
    dataset_type: str = "annotable",
    batch_size: int = Query(32, ge=1, le=1024),
    external_dataset: TextDatasetModel | None = None,
) -> None:
    """
    Start prediction with a model
    - quick or bert model
    - types of dataset
    Manage specific cases for prediction

    TODO : optimize prediction on whole dataset
    TODO : manage prediction external/whole dataset for quick models

    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        # types of prediction
        if kind not in ["quick", "bert", "image", "ner"]:
            raise InvalidInputError(f"Model kind {kind} not recognized")

        if dataset_type not in ["annotable", "external", "all"]:
            raise InvalidInputError(f"Dataset {dataset_type} not recognized")

        # the external dataset arrives as a staged chunked upload (see
        # activetigger.uploads): move it into the project data folder where
        # the prediction task reads it back by filename
        if external_dataset is not None:
            target = get_upload_staging().move_to(
                current_user.username,
                external_dataset.upload_id,
                project.data.path_datasets,
                (".csv", ".parquet", ".xlsx"),
            )
            external_dataset.filename = target.name

        # managing the perimeter of the prediction
        datasets = None
        if dataset_type == "annotable":
            datasets = ["train"]
            if project.data.valid is not None:
                datasets.append("valid")
            if project.data.test is not None:
                datasets.append("test")
        if dataset_type == "external":
            if kind not in ("bert", "quick"):
                raise Exception(
                    "External dataset prediction is only available for bert and quick models"
                )

        if kind == "bert":
            if dataset_type == "external" and external_dataset is None:
                raise Exception("External dataset must be provided for external prediction")
            project.start_language_model_prediction(
                username=current_user.username,
                dataset_type=dataset_type,
                datasets=datasets,
                scheme_name=scheme,
                model_name=model_name,
                external_dataset=external_dataset,
                batch_size=batch_size,
            )

        if kind == "quick":
            if dataset_type == "all":
                # Predict on the whole source dataset (path_data_all) — the
                # PredictWithFeatures task (re)computes the features the
                # model was trained on and applies the model in one go.
                project.quickmodels.predict_on_dataset(
                    name=model_name,
                    username=current_user.username,
                    dataset="all",
                )
            elif dataset_type == "external":
                if external_dataset is None:
                    raise Exception("External dataset must be provided for external prediction")
                project.quickmodels.predict_on_external_dataset(
                    name=model_name,
                    username=current_user.username,
                    external_dataset=external_dataset,
                )
            else:
                if datasets is None:
                    raise Exception(
                        "Dataset parameter must be specified for quick model prediction"
                    )
                project.start_quick_model_prediction(
                    username=current_user.username,
                    dataset_type=dataset_type,
                    datasets=datasets,
                    scheme_name=scheme,
                    model_name=model_name,
                )

        if kind == "image":
            project.start_image_model_prediction(
                username=current_user.username,
                dataset_type=dataset_type,
                datasets=datasets,
                scheme_name=scheme,
                model_name=model_name,
                batch_size=batch_size,
            )

        if kind == "ner":
            project.start_ner_prediction(
                username=current_user.username,
                dataset_type=dataset_type,
                datasets=datasets,
                scheme_name=scheme,
                model_name=model_name,
                external_dataset=external_dataset,
                batch_size=batch_size,
            )
        get_orchestrator().log_action(
            current_user.username,
            f"PREDICT MODEL: {model_name} - {kind} DATASET: {dataset_type}",
            project.name,
        )
    except Exception as e:
        print(f"ERROR in /models/predict\n{e}\n\n")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/train", dependencies=[Depends(verified_user)])
def post_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert: BertModelModel,
) -> None:
    """
    Compute bertmodel
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    if getattr(project.params, "kind", "text") == "image":
        raise HTTPException(
            status_code=400, detail="BERT models are not supported for image projects"
        )
    try:
        orchestrator = get_orchestrator()
        if not orchestrator.available_storage(current_user.username):
            raise HTTPException(
                status_code=403,
                detail="Storage limit exceeded. Please delete models or contact the administrator.",
            )
        project.start_languagemodel_training(
            bert=bert,
            username=current_user.username,
        )
        orchestrator.log_action(current_user.username, f"TRAIN MODEL: {bert.name}", project.name)
        return None

    except Exception as e:
        print(f"ERREUR : \n{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/delete", dependencies=[Depends(verified_user)])
def delete_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert_name: ModelName,
) -> None:
    """
    Delete trained bert model
    # TODO : check the replace
    """
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    if getattr(project.params, "kind", "text") == "image":
        raise HTTPException(
            status_code=400, detail="BERT models are not supported for image projects"
        )
    try:
        # delete the model
        project.languagemodels.delete(bert_name)
        # delete the features associated with the model
        for f in [i for i in project.features.map.keys() if bert_name.replace("__", "_") in i]:
            project.features.delete(f)
        get_orchestrator().log_action(
            current_user.username, f"DELETE MODEL + FEATURES: {bert_name}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/rename", dependencies=[Depends(verified_user)])
def rename_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: ModelName,
    new_name: ModelName,
) -> None:
    """
    Rename bertmodel
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    if getattr(project.params, "kind", "text") == "image":
        raise HTTPException(
            status_code=400, detail="BERT models are not supported for image projects"
        )
    try:
        project.languagemodels.rename(former_name, new_name)
        get_orchestrator().log_action(
            current_user.username,
            f"INFO RENAME MODEL: {former_name} -> {new_name}",
            project.name,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/image/train", dependencies=[Depends(verified_user)])
def post_image(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    image: ImageModelModel,
) -> None:
    """
    Fine-tune an image-classification model on an image project.
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    if getattr(project.params, "kind", "text") != "image":
        raise HTTPException(
            status_code=400, detail="Image models are only supported for image projects"
        )
    try:
        orchestrator = get_orchestrator()
        if not orchestrator.available_storage(current_user.username):
            raise HTTPException(
                status_code=403,
                detail="Storage limit exceeded. Please delete models or contact the administrator.",
            )
        project.start_image_model_training(image=image, username=current_user.username)
        orchestrator.log_action(
            current_user.username, f"TRAIN IMAGE MODEL: {image.name}", project.name
        )
        return None
    except Exception as e:
        print(f"ERROR /models/image/train: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/image/delete", dependencies=[Depends(verified_user)])
def delete_image(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    image_name: ModelName,
) -> None:
    """
    Delete a trained image-classification model + its derived features.
    """
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    if getattr(project.params, "kind", "text") != "image":
        raise HTTPException(
            status_code=400, detail="Image models are only supported for image projects"
        )
    if project.imagemodels is None:
        raise HTTPException(status_code=400, detail="Image manager not initialized")
    try:
        project.imagemodels.delete(image_name)
        for f in [i for i in project.features.map.keys() if image_name.replace("__", "_") in i]:
            project.features.delete(f)
        get_orchestrator().log_action(
            current_user.username, f"DELETE IMAGE MODEL + FEATURES: {image_name}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/ner/train", dependencies=[Depends(verified_user)])
def post_ner(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    ner: NerModelModel,
) -> None:
    """
    Fine-tune a token-classification model on a span scheme.
    Experimental feature — gated in the frontend by developmentMode.
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    if getattr(project.params, "kind", "text") == "image":
        raise HTTPException(
            status_code=400, detail="NER models are not supported for image projects"
        )
    try:
        orchestrator = get_orchestrator()
        if not orchestrator.available_storage(current_user.username):
            raise HTTPException(
                status_code=403,
                detail="Storage limit exceeded. Please delete models or contact the administrator.",
            )
        project.start_ner_training(ner=ner, username=current_user.username)
        orchestrator.log_action(current_user.username, f"TRAIN NER MODEL: {ner.name}", project.name)
        return None
    except Exception as e:
        print(f"ERROR /models/ner/train: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/ner/delete", dependencies=[Depends(verified_user)])
def delete_ner(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    ner_name: ModelName,
) -> None:
    """
    Delete a trained NER model and any features derived from it.
    """
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    if project.nermodels is None:
        raise HTTPException(status_code=400, detail="NER models are not available for this project")
    try:
        project.nermodels.delete(ner_name)
        for f in [i for i in project.features.map.keys() if ner_name.replace("__", "_") in i]:
            project.features.delete(f)
        get_orchestrator().log_action(
            current_user.username, f"DELETE NER MODEL: {ner_name}", project.name
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/ner/rename", dependencies=[Depends(verified_user)])
def rename_ner(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: ModelName,
    new_name: ModelName,
) -> None:
    """
    Rename a NER model.
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    if project.nermodels is None:
        raise HTTPException(status_code=400, detail="NER models are not available for this project")
    try:
        project.nermodels.rename(former_name, new_name)
        get_orchestrator().log_action(
            current_user.username,
            f"INFO RENAME NER MODEL: {former_name} -> {new_name}",
            project.name,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/image/rename", dependencies=[Depends(verified_user)])
def rename_image(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: ModelName,
    new_name: ModelName,
) -> None:
    """
    Rename an image-classification model.
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    if getattr(project.params, "kind", "text") != "image":
        raise HTTPException(
            status_code=400, detail="Image models are only supported for image projects"
        )
    if project.imagemodels is None:
        raise HTTPException(status_code=400, detail="Image manager not initialized")
    try:
        project.imagemodels.rename(former_name, new_name)
        get_orchestrator().log_action(
            current_user.username,
            f"INFO RENAME IMAGE MODEL: {former_name} -> {new_name}",
            project.name,
        )
    except (HTTPException, APIError, OverflowError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
