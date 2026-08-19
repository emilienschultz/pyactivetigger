import csv
import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from activetigger.datamodels import ProjectBaseModel, ProjectModel
from activetigger.functions import concat_text_columns, slugify
from activetigger.functions_image import (
    ALLOWED_EXT as IMAGE_ALLOWED_EXT,
)
from activetigger.functions_image import (
    MAX_IMAGE_BYTES,
    MAX_ZIP_BYTES,
    filter_readable_images,
    generate_thumbnail,
)
from celery import Task

# from celery.utils.log import get_task_logger
from pydantic import BaseModel

from task_manager.auto_callback_task import AutoCallbackTask, QueueName
from task_manager.celery import celery_app

# TODO: is that necessary? we don't use csv lib in this code
csv.field_size_limit(sys.maxsize)

# Return type must be a dict, a BaseModel would not be serialized by Celery
class CreateProjectTaskResult(TypedDict):
    username: str
    project: dict[str, Any] # Dumped Project model
    import_trainset_path: str | None
    import_testset_path: str | None
    import_validset_path: str | None


# input can be a BaseModel if pydantic is enabled in the Task decorator
class CreateProjectTaskInput(BaseModel):
    image_project: bool
    username: str
    project_slug: str
    params: ProjectBaseModel
    data_all: str
    train_file: str
    valid_file: str
    test_file: str
    features_file: str
    random_seed: int


# Task definition using the auto callback generic parent task class
class CreateProjectTask(AutoCallbackTask):
    name = "create project"
    queue = QueueName.CPU

    def create_project(self: Task, props: CreateProjectTaskInput) -> CreateProjectTaskResult:
        """
        Create the project with the given name and file
        Define an internal and external index
        return the project model and the train/valid/test datasets to import in the database
        """
        # TODO us task logger
        # logger = get_task_logger(CreateProjectTask.name)
        print(f"Start queue project {props.project_slug} for {props.username}")
        # check if the directory already exists + file (should with the data)
        if props.params.dir is None or not props.params.dir.exists():
            print("The directory does not exist and should", props.params.dir)
            raise Exception("The directory does not exist and should")

        # Step 1 : load all data, rename columns and define index
        processed_corpus = False
        if props.params.filename is None or props.params.from_toy_dataset:
            processed_corpus = True

        if (props.params.filename is not None) and not props.params.from_toy_dataset:
            # if a file was uploaded
            # load the uploaded file
            file_path = props.params.dir.joinpath(props.params.filename)
        else:
            # if the file was copied from a toy dataset / existing project
            file_path = props.params.dir.joinpath(props.data_all)

        if not file_path.exists():
            raise Exception("File not found, problem when uploading")

        if str(file_path).endswith(".csv"):
            try:
                content = pd.read_csv(
                    file_path, sep=None, low_memory=False, on_bad_lines="skip", engine="python"
                )
            except Exception:
                content = pd.read_csv(file_path, sep=None, on_bad_lines="skip", engine="python")
        elif str(file_path).endswith(".parquet"):
            content = pd.read_parquet(file_path)
        elif str(file_path).endswith(".xlsx"):
            content = pd.read_excel(file_path)
        else:
            raise Exception("File format not supported (only csv, xlsx and parquet)")

        # if no already processed
        if not processed_corpus:
            # rename columns both for data & params to avoid confusion (use internal dataset_ prefix)
            content.columns = ["dataset_" + i for i in content.columns]
            if props.params.col_id is not None:
                props.params.col_id = "dataset_" + props.params.col_id

            # change also the name in the parameters
            props.params.cols_text = ["dataset_" + i for i in props.params.cols_text if i]
            props.params.cols_context = ["dataset_" + i for i in props.params.cols_context if i]
            props.params.cols_label = ["dataset_" + i for i in props.params.cols_label if i]
            props.params.cols_stratify = ["dataset_" + i for i in props.params.cols_stratify if i]
            if props.params.col_split:
                props.params.col_split = "dataset_" + props.params.col_split

            # remove completely empty lines
            content = content.dropna(how="all")

        # quickfix : case where the index is the row number
        if props.params.col_id == "row_number":
            props.params.col_id = "dataset_row_number"

        all_columns = list(content.columns)
        n_total = len(content)

        # if using a column to define the split, validate it exists; size checks
        # below don't apply because counts come from the column values themselves
        if props.params.col_split:
            if props.params.col_split not in content.columns:
                shutil.rmtree(props.params.dir)
                raise Exception(
                    f"Split column '{props.params.col_split.removeprefix('dataset_')}' "
                    f"not found in the dataset."
                )
        # test if the size of the sample requested is possible
        elif n_total < props.params.n_test + props.params.n_valid + props.params.n_train:
            shutil.rmtree(props.params.dir)
            raise Exception(
                f"Not enough data for creating the train/valid/test dataset. Current : {len(content)} ; Selected : {props.params.n_test + props.params.n_valid + props.params.n_train}"
            )

        # create the internal/external index
        # case where the index is the row number
        if props.params.col_id == "dataset_row_number":
            content["id_internal"] = [str(i) for i in range(len(content))]
            content["id_external"] = content["id_internal"]

        # case the index is a column: check uniqueness then use slugify for internal if unique after slugify
        else:
            content["id_external"] = content[props.params.col_id].astype(str)
            if content["id_external"].nunique() != len(content):
                n_duplicates = len(content) - content["id_external"].nunique()
                raise Exception(
                    f"The selected ID column '{props.params.col_id.removeprefix('dataset_')}' "
                    f"contains {n_duplicates} duplicate values. "
                    f"Please choose a column with unique values or use 'Row number'."
                )
            col_slugified = content["id_external"].apply(slugify)
            if col_slugified.nunique() == len(content):
                content["id_internal"] = col_slugified
            else:
                content["id_internal"] = [str(i) for i in range(len(content))]

        content.set_index("id_internal", inplace=True)

        # convert columns that can be numeric or force text, exception for the text/labels
        for col in [i for i in content.columns if i not in props.params.cols_label]:
            try:
                content[col] = pd.to_numeric(content[col], errors="raise")
            except Exception:
                content[col] = content[col].astype(str).replace("nan", None)
        for col in props.params.cols_label:
            try:
                content[col] = content[col].astype(str).replace("nan", None)
            except Exception:
                # if the column is not convertible to string, keep it as is
                pass

        # create the text column, merging the different columns
        content["text"] = concat_text_columns(content, props.params.cols_text)

        # convert NA texts in empty string
        content["text"] = content["text"].fillna("")

        # save a complete copy of the dataset
        content.to_parquet(props.params.dir.joinpath(props.data_all), index=True)

        # ------------------------
        # End of the data cleaning
        # ------------------------
        # Step 2 : valid/test dataset : from the complete dataset + random/stratification
        # if both, draw together valid / test set and then separage
        rows_test = []
        rows_valid = []
        props.params.test = False
        props.params.valid = False
        testset: pd.DataFrame | None = None
        validset: pd.DataFrame | None = None
        trainset: pd.DataFrame | None = None

        # Column-based split: rows are assigned by the value of col_split
        # (only "train", "valid", "test" are honored — everything else is dropped).
        if props.params.col_split:
            split_values = content[props.params.col_split].astype(str).str.strip().str.lower()
            trainset = content[split_values == "train"]
            validset_candidate = content[split_values == "valid"]
            testset_candidate = content[split_values == "test"]

            if len(testset_candidate) > 0:
                testset = testset_candidate
                testset.to_parquet(props.params.dir.joinpath(props.test_file), index=True)
                props.params.test = True
                rows_test = list(testset.index)
                props.params.n_test = len(testset)
            if len(validset_candidate) > 0:
                validset = validset_candidate
                validset.to_parquet(props.params.dir.joinpath(props.valid_file), index=True)
                props.params.valid = True
                rows_valid = list(validset.index)
                props.params.n_valid = len(validset)
            props.params.n_train = len(trainset)
            if len(trainset) == 0:
                shutil.rmtree(props.params.dir)
                raise Exception(
                    f"No rows with value 'train' in column "
                    f"'{props.params.col_split.removeprefix('dataset_')}'."
                )
            trainset[["id_external", "text"] + props.params.cols_context].to_parquet(
                props.params.dir.joinpath(props.train_file), index=True
            )

        # case there is a test or valid set, common draw
        elif props.params.n_test + props.params.n_valid != 0:
            n_to_draw = props.params.n_test + props.params.n_valid
            # manage stratification
            if len(props.params.cols_stratify) == 0 or not props.params.stratify_eval:
                draw = content.sample(n_to_draw, random_state=props.random_seed)
            else:
                df_grouped = content.groupby(props.params.cols_stratify, group_keys=False)
                nb_cat = len(df_grouped)
                nb_elements_cat = round(n_to_draw / nb_cat)
                sampled_idx = df_grouped.apply(
                    lambda x: x.sample(min(len(x), nb_elements_cat), random_state=props.random_seed)
                ).index.get_level_values(-1)
                draw = content.loc[sampled_idx]

            # divide between test and valid
            if props.params.n_test > 0 and props.params.n_valid == 0:
                testset = draw
                testset.to_parquet(props.params.dir.joinpath(props.test_file), index=True)
                props.params.test = True
                rows_test = list(testset.index)
            elif props.params.n_valid > 0 and props.params.n_test == 0:
                validset = draw
                validset.to_parquet(props.params.dir.joinpath(props.valid_file), index=True)
                props.params.valid = True
                rows_valid = list(validset.index)
            else:
                testset = draw.sample(props.params.n_test, random_state=props.random_seed)
                validset = draw.drop(index=testset.index)
                validset.to_parquet(props.params.dir.joinpath(props.valid_file), index=True)
                testset.to_parquet(props.params.dir.joinpath(props.test_file), index=True)
                props.params.test = True
                props.params.valid = True
                rows_valid = list(validset.index)
                rows_test = list(testset.index)

        # Step 3 : train dataset / different strategies
        # Skipped when col_split is set — train rows were already chosen by the column.
        if not props.params.col_split:
            # remove test rows
            content = content.drop(rows_test)
            content = content.drop(rows_valid)

            # case where there is no valid/test set and the selection is deterministic
            if (
                not props.params.random_selection
                and props.params.n_test == 0
                and props.params.n_valid == 0
            ):
                trainset = content[0 : props.params.n_train]
            # case to force the max of label from all label columns
            elif props.params.force_label and len(props.params.cols_label) > 0:
                f_notna = content[content[props.params.cols_label].notna().any(axis=1)]
                f_na = content[content[props.params.cols_label].isna().all(axis=1)]
                # different case regarding the number of labels
                if len(f_notna) > props.params.n_train:
                    trainset = f_notna.sample(props.params.n_train, random_state=props.random_seed)
                else:
                    n_train_random = props.params.n_train - len(f_notna)
                    trainset = pd.concat(
                        [
                            f_notna,
                            f_na.sample(n_train_random, random_state=props.random_seed),
                        ]
                    )
            # case there is stratification on the trainset
            elif len(props.params.cols_stratify) > 0 and props.params.stratify_train:
                df_grouped = content.groupby(props.params.cols_stratify, group_keys=False)
                nb_cat = len(df_grouped)
                nb_elements_cat = round(props.params.n_train / nb_cat)
                sampled_idx = df_grouped.apply(
                    lambda x: x.sample(min(len(x), nb_elements_cat), random_state=props.random_seed)
                ).index.get_level_values(-1)
                trainset = content.loc[sampled_idx]
            # default with random selection in the remaining elements
            else:
                trainset = content.sample(props.params.n_train, random_state=props.random_seed)

            # write the trainset
            trainset[["id_external", "text"] + props.params.cols_context].to_parquet(
                props.params.dir.joinpath(props.train_file), index=True
            )


        # add elements for the parameters
        props.params.n_total= n_total
        project_to_create = ProjectModel(
            project_slug=props.project_slug,
            all_columns=all_columns,
            **props.params.model_dump()
        )
        

        # schemes/labels to import (in the main process)
        import_trainset_path = None
        import_testset_path = None
        import_validset_path = None
        if len(props.params.cols_label) > 0 and trainset is not None:
            import_trainset = trainset[props.params.cols_label].dropna(how="all")
            # write panda file on disk
            import_trainset_path = props.params.dir.joinpath("import_train.parquet")
            import_trainset.to_parquet(import_trainset_path, index=True)
            if testset is not None and not props.params.clear_test:
                import_testset = testset[props.params.cols_label].dropna(how="all")
                import_testset_path = props.params.dir.joinpath("import_test.parquet")
                import_testset.to_parquet(import_testset_path, index=True)
            if validset is not None and not props.params.clear_valid:
                import_validset = validset[props.params.cols_label].dropna(how="all")
                import_validset_path = props.params.dir.joinpath("import_valid.parquet")
                import_validset.to_parquet(import_validset_path, index=True)


        # delete the initial file
        if props.params.filename is not None:
            try:
                props.params.dir.joinpath(props.params.filename).unlink()
            except OSError as e:
                print(f"Warning: could not delete uploaded file: {e}")

        result= CreateProjectTaskResult(
            username= props.username,
            project= project_to_create.model_dump(mode='json'),
            import_trainset_path= str(import_trainset_path) if import_trainset_path else None,
            import_testset_path= str(import_testset_path) if import_testset_path else None,
            import_validset_path= str(import_validset_path) if import_validset_path else None,
        )
        return result

    # TODO: refacto those two sibblings method by factoring code between image/normal project
    def create_image_project(self: Task, props: CreateProjectTaskInput) -> CreateProjectTaskResult:
        """
        Experimental task for creating an "image" project.
        Accepts a .zip upload containing png/jpg files and an optional metadata
        csv/parquet. Builds data_all.parquet with id/path (+ metadata) columns.
        Lives next to CreateProject to be easy to grep and eventually remove.
        """

        print(f"Start queue image project {props.project_slug} for {props.username}")

        if props.params.dir is None or not props.params.dir.exists():
            raise Exception("The directory does not exist and should")

        if props.params.filename is None:
            raise Exception("An image .zip archive must be uploaded for image projects")

        zip_path = props.params.dir.joinpath(props.params.filename)
        if not zip_path.exists() or not str(zip_path).lower().endswith(".zip"):
            raise Exception("Image project expects a .zip archive upload")

        # cap on zip size
        if zip_path.stat().st_size > MAX_ZIP_BYTES:
            raise Exception(f"Zip file too large (max {MAX_ZIP_BYTES // (1024 * 1024)} MB)")

        # read central directory first, validate, then extract
        images_dir = props.params.dir.joinpath("images")
        images_dir.mkdir(parents=True, exist_ok=True)
        thumbs_dir = images_dir.joinpath("thumbs")
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            image_infos = [
                i
                for i in infos
                if Path(i.filename).suffix.lower() in IMAGE_ALLOWED_EXT
                and not Path(i.filename).name.startswith("._")
                and "__MACOSX" not in Path(i.filename).parts
            ]
            if len(image_infos) == 0:
                raise Exception("Zip does not contain any .png/.jpg image")
            for info in image_infos:
                if info.file_size > MAX_IMAGE_BYTES:
                    raise Exception(
                        f"Image '{info.filename}' exceeds per-file cap of "
                        f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB"
                    )

            # optional metadata file in the zip
            metadata_infos = [
                i for i in infos if Path(i.filename).suffix.lower() in {".csv", ".parquet"}
            ]

            total_images = len(image_infos)
            progress_file = props.params.dir.joinpath("creation_progress")
            for idx, info in enumerate(image_infos, 1):
                src_name = Path(info.filename).name
                element_id = Path(info.filename).stem
                target = images_dir.joinpath(src_name)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                rows.append({"id": element_id, "path": str(target)})

                # Precompute a 256px JPEG thumbnail keyed by the slugified id
                # (id_internal). Failures on individual files are logged and
                # skipped — the thumbnail route falls back to the original.
                generate_thumbnail(target, thumbs_dir.joinpath(f"{slugify(element_id)}.jpg"))

                # Write progress
                progress_pct = round((idx / total_images) * 100, 1)
                try:
                    with open(progress_file, "w") as pf:
                        pf.write(str(progress_pct))
                except OSError:
                    pass

            # Clean up progress file
            try:
                progress_file.unlink()
            except OSError:
                pass

            metadata_df = None
            if metadata_infos:
                meta_info = metadata_infos[0]
                tmp_meta = props.params.dir.joinpath(Path(meta_info.filename).name)
                with zf.open(meta_info) as src, open(tmp_meta, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                try:
                    if str(tmp_meta).endswith(".csv"):
                        metadata_df = pd.read_csv(tmp_meta)
                    else:
                        metadata_df = pd.read_parquet(tmp_meta)
                except Exception as e:
                    print(f"Could not read metadata file: {e}")
                    metadata_df = None
                try:
                    tmp_meta.unlink()
                except OSError:
                    pass

        # Drop unreadable / corrupt images now, so the train pool only
        # contains images the rest of the pipeline can actually load.
        if rows:
            readable_flags = filter_readable_images([r["path"] for r in rows])
            n_dropped = sum(1 for f in readable_flags if not f)
            if n_dropped > 0:
                kept = []
                for row, ok in zip(rows, readable_flags):
                    if ok:
                        kept.append(row)
                    else:
                        try:
                            Path(row["path"]).unlink(missing_ok=True)
                        except OSError:
                            pass
                rows = kept
                print(f"Dropped {n_dropped} unreadable/corrupt images", flush=True)
        if not rows:
            raise Exception("Zip contained no readable images after cleaning")

        content = pd.DataFrame(rows)
        if metadata_df is not None and len(metadata_df) > 0:
            # match on filename stem; first column of metadata is assumed to be the id
            id_col = metadata_df.columns[0]
            metadata_df[id_col] = metadata_df[id_col].astype(str).map(lambda s: Path(s).stem)
            metadata_df = metadata_df.rename(columns={id_col: "id"})
            content = content.merge(metadata_df, on="id", how="left")

        # internal / external indexing to match CreateProject expectations
        content["id_external"] = content["id"].astype(str)
        content["id_internal"] = content["id_external"].apply(slugify)
        if content["id_internal"].nunique() != len(content):
            content["id_internal"] = [str(i) for i in range(len(content))]
        content.set_index("id_internal", inplace=True)
        # the rest of the pipeline expects a "text" column
        content["text"] = content["path"].astype(str)

        all_columns = list(content.columns)
        n_total = len(content)

        # simple split: everything is train by default; honor n_test / n_valid if set
        n_test = max(0, int(props.params.n_test or 0))
        n_valid = max(0, int(props.params.n_valid or 0))
        n_train = int(props.params.n_train or (n_total - n_test - n_valid))
        n_train = max(0, min(n_train, n_total - n_test - n_valid))

        # Eval images are duplicated into images/eval_{dataset}/ so the user
        # can drop the test/valid set later (rmtree of the subdir) without
        # touching the originals that the train pool still references.
        def _copy_eval_images(sampled: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
            eval_dir = images_dir.joinpath(f"eval_{dataset_name}")
            eval_dir.mkdir(parents=True, exist_ok=True)
            new_paths: list[str] = []
            for orig in sampled["path"].astype(str):
                src = Path(orig)
                dst = eval_dir.joinpath(src.name)
                if dst.exists():
                    dst = eval_dir.joinpath(f"{uuid.uuid4().hex[:6]}_{src.name}")
                shutil.copy2(src, dst)
                new_paths.append(str(dst))
            out = sampled.copy()
            out["path"] = new_paths
            out["text"] = new_paths
            return out

        testset = None
        validset = None
        rows_test: list = []
        rows_valid: list = []
        props.params.test = False
        props.params.valid = False
        if n_test + n_valid > 0:
            draw = content.sample(n_test + n_valid, random_state=props.random_seed)
            if n_test > 0:
                testset = draw.sample(n_test, random_state=props.random_seed)
                rows_test = list(testset.index)
                testset = _copy_eval_images(testset, "test")
                testset.to_parquet(props.params.dir.joinpath(props.test_file), index=True)
                props.params.test = True
            if n_valid > 0:
                validset = draw.drop(index=rows_test) if rows_test else draw
                validset = validset.sample(n_valid, random_state=props.random_seed)
                rows_valid = list(validset.index)
                validset = _copy_eval_images(validset, "valid")
                validset.to_parquet(props.params.dir.joinpath(props.valid_file), index=True)
                props.params.valid = True

        remaining = content.drop(rows_test + rows_valid, errors="ignore")
        trainset = (
            remaining.sample(min(n_train, len(remaining)), random_state=props.random_seed)
            if n_train > 0
            else remaining
        )

        # persist data_all
        content.to_parquet(props.params.dir.joinpath(props.data_all), index=True)
        # persist train
        trainset[["id_external", "text"]].to_parquet(
            props.params.dir.joinpath(props.train_file), index=True
        )

        # NB: image embeddings are NOT computed here. They can be computed
        # on-demand later via the features API.

        # make sure cols_text has the expected shape for downstream code
        props.params.cols_text = ["text"]
        props.params.col_id = "id"



        # add elements for the parameters
        props.params.n_total= n_total
        project_to_create = ProjectModel(
            project_slug=props.project_slug,
            all_columns=all_columns,
            **props.params.model_dump()
        )
        

        # delete uploaded zip
        try:
            zip_path.unlink()
        except OSError as e:
            print(f"Warning: could not delete uploaded zip: {e}")

    

        result= CreateProjectTaskResult(
            username= props.username,
            project= project_to_create.model_dump(mode='json'),
            import_trainset_path= None,
            import_testset_path= None,
            import_validset_path= None,
        )
        return result

# Task registration
@celery_app.task(
    bind=True,
    name=CreateProjectTask.name,
    queue=CreateProjectTask.queue,
    base=CreateProjectTask,
    pydantic=True,
)
def create_project_task(self, props: CreateProjectTaskInput):
    if props.image_project:
        return CreateProjectTask.create_image_project(self=self, props=props)
    else:
        return CreateProjectTask.create_project(self=self, props=props)
