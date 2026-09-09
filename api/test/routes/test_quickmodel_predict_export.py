"""
End-to-end check for the quickmodel predict-on-complete-dataset -> export chain
(diagnostic script, run with pytest).
"""

import time

from fastapi.testclient import TestClient

from test.utils import (
    add_label,
    annotate_element,
    create_project,
    get_next_element_id,
    get_project_state,
)

TIMEOUT = 120


def wait_no_processes(client, headers, slug, timeout=TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        r = client.get(f"/api/projects/{slug}", headers=headers)
        assert r.status_code == 200, r.text
        state = r.json()
        training = state.get("quickmodel", {}).get("training", {})
        features_training = state.get("features", {}).get("training", {})
        if not training and not features_training:
            return state
        time.sleep(1)
    raise TimeoutError("processes still running")


def test_quickmodel_predict_all_export(client: TestClient, superuser_headers: dict[str, str]):
    project = create_project(client, superuser_headers, f"Quick-{int(time.time())}")
    slug = project["project_slug"]

    # use the default scheme created with the project
    state = get_project_state(client, superuser_headers, slug)
    scheme = list(state["schemes"]["available"].keys())[0]
    print("scheme:", scheme)

    add_label(client, superuser_headers, slug, scheme, "label0")
    add_label(client, superuser_headers, slug, scheme, "label1")

    # annotate 8 elements alternating two labels
    history: list[str] = []
    for i in range(8):
        eid = get_next_element_id(client, superuser_headers, slug, scheme, history)
        history.append(eid)
        annotate_element(client, superuser_headers, slug, scheme, eid, f"label{i % 2}")

    # compute a dfm feature
    r = client.post(
        f"/api/features/add?project_slug={slug}",
        json={
            "type": "dfm",
            "name": "dfm",
            "use_default_name": True,
            "parameters": {
                "dfm_tfidf": False,
                "ngrams": 1,
                "min_term_freq": 0.01,
                "max_term_freq": 1.0,
            },
        },
        headers=superuser_headers,
    )
    assert r.status_code == 200, r.text
    state = wait_no_processes(client, superuser_headers, slug)
    start = time.time()
    features: list = []
    while time.time() - start < 60:
        state = get_project_state(client, superuser_headers, slug)
        features = list(state["features"]["available"])
        if features:
            break
        time.sleep(1)
    print("features:", features)
    print("features training:", state["features"].get("training"))
    print("errors:", state.get("errors"))
    assert features

    # also compute a deterministic regex feature to compare behaviours
    r = client.post(
        f"/api/features/add?project_slug={slug}",
        json={
            "type": "regex",
            "name": "hasthe",
            "use_default_name": False,
            "parameters": {"value": "the"},
        },
        headers=superuser_headers,
    )
    assert r.status_code == 200, r.text
    start = time.time()
    regex_feature = None
    while time.time() - start < 60:
        state = get_project_state(client, superuser_headers, slug)
        regex_feature = next((f for f in state["features"]["available"] if f not in features), None)
        if regex_feature:
            break
        time.sleep(1)
    print("regex feature:", regex_feature, "| errors:", state.get("errors"))
    assert regex_feature

    # train a quickmodel
    r = client.post(
        f"/api/models/quick/train?project_slug={slug}",
        json={
            "name": "qm-test",
            "scheme": scheme,
            "model": "logistic-l2",
            "features": [regex_feature],
            "params": {"costLogL2": 1},
            "cv10": False,
        },
        headers=superuser_headers,
    )
    assert r.status_code == 200, r.text
    start = time.time()
    avail: dict = {}
    while time.time() - start < 60:
        state = get_project_state(client, superuser_headers, slug)
        avail = state["quickmodel"]["available"]
        if scheme in avail and any(m["name"] == "qm-test" for m in avail[scheme]):
            break
        time.sleep(1)
    print("quickmodel available after train:", avail)
    print("errors:", state.get("errors"))
    assert scheme in avail and any(m["name"] == "qm-test" for m in avail[scheme])

    # predict on the complete dataset
    r = client.post(
        f"/api/models/predict?project_slug={slug}"
        f"&model_name=qm-test&scheme={scheme}&kind=quick&dataset_type=all",
        headers=superuser_headers,
    )
    assert r.status_code == 200, r.text
    state = wait_no_processes(client, superuser_headers, slug)

    # poll the state for the predicted_all flag (state cache ~2s)
    flag = False
    row = None
    start = time.time()
    while time.time() - start < 15:
        r = client.get(f"/api/projects/{slug}", headers=superuser_headers)
        rows = r.json()["quickmodel"]["available"].get(scheme, [])
        row = next((m for m in rows if m["name"] == "qm-test"), None)
        flag = bool(row and row.get("predicted_all"))
        if flag:
            break
        time.sleep(1)
    print("model row after predict:", row)

    # check the file exists on disk
    from pathlib import Path

    from activetigger.config import config

    pred = (
        Path(config.data_path)
        / "projects"
        / slug
        / "quickmodels"
        / "qm-test"
        / "predict_all.parquet"
    )
    print("prediction file exists:", pred.exists(), pred)

    # try the export endpoint directly
    r = client.get(
        f"/api/export/prediction?project_slug={slug}"
        f"&format=csv&name=qm-test&dataset=all&kind=quick",
        headers=superuser_headers,
    )
    print("export status:", r.status_code, r.text[:300] if r.status_code != 200 else "OK")

    assert pred.exists(), "prediction file missing on disk"
    assert flag, f"predicted_all flag not set in state; row={row}"
    assert r.status_code == 200
