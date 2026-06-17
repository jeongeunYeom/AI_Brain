from app.services.jobs import create_job, get_job, update_job


def test_upload_job_status_lifecycle():
    job_id = create_job()
    update_job(job_id, 2, "[2/6] 텍스트 추출 중")
    running = get_job(job_id)

    assert running is not None
    assert running["step"] == 2
    assert running["status"] == "running"

    update_job(job_id, 6, "Processed 1222 chunks", status="completed")
    completed = get_job(job_id)

    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["message"] == "Processed 1222 chunks"
