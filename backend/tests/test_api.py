from fastapi import HTTPException

from sentinel.main import experiments, experiments_csv, health, parse_sweep_strengths


def test_health_endpoint_reports_runtime_without_loading_model():
    payload = health()
    assert payload["status"] == "ready"
    assert "device" in payload and "model_loaded" in payload


def test_experiment_list_is_empty_or_returns_saved_rows():
    assert isinstance(experiments(), list)


def test_sweep_rejects_invalid_strengths_without_loading_model():
    try:
        parse_sweep_strengths("0,0.3,not-a-number")
    except HTTPException as exc:
        assert exc.status_code == 422
        assert "comma-separated numbers" in exc.detail
    else:
        raise AssertionError("invalid strengths should be rejected")


def test_csv_export_includes_experiment_details():
    response = experiments_csv()
    assert response.status_code == 200
    assert b"details" in response.body.splitlines()[0]
