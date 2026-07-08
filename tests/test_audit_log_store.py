from backend import audit_log_store


def setup_function():
    audit_log_store.clear_audit_events()


def test_create_audit_event_creates_sequential_ids():
    first = audit_log_store.create_audit_event(audit_log_store.AGENT_RUN_CREATED, "created")
    second = audit_log_store.create_audit_event(audit_log_store.APPROVAL_REQUIRED, "required")

    assert first["audit_id"] == "AUDIT-0001"
    assert second["audit_id"] == "AUDIT-0002"


def test_list_audit_events_filters_by_event_type_run_id_and_case_id():
    audit_log_store.create_audit_event(audit_log_store.AGENT_RUN_CREATED, "created", run_id="RUN-1")
    audit_log_store.create_audit_event(audit_log_store.CASE_CREATED, "case", run_id="RUN-1", case_id="CASE-1")
    audit_log_store.create_audit_event(audit_log_store.APPROVAL_REJECTED, "reject", run_id="RUN-2")

    assert [event["event_type"] for event in audit_log_store.list_audit_events(event_type=audit_log_store.CASE_CREATED)] == [
        audit_log_store.CASE_CREATED
    ]
    assert len(audit_log_store.list_audit_events(run_id="RUN-1")) == 2
    assert audit_log_store.list_audit_events(case_id="CASE-1")[0]["event_type"] == audit_log_store.CASE_CREATED


def test_limit_returns_most_recent_events():
    audit_log_store.create_audit_event("ONE", "one")
    audit_log_store.create_audit_event("TWO", "two")
    audit_log_store.create_audit_event("THREE", "three")

    events = audit_log_store.list_audit_events(limit=2)

    assert [event["event_type"] for event in events] == ["TWO", "THREE"]


def test_get_audit_event_returns_copy_and_missing_returns_none():
    created = audit_log_store.create_audit_event(audit_log_store.AGENT_RUN_CREATED, "created")
    fetched = audit_log_store.get_audit_event(created["audit_id"])
    fetched["metadata"]["status"] = "mutated"

    assert audit_log_store.get_audit_event(created["audit_id"])["metadata"] == {}
    assert audit_log_store.get_audit_event("AUDIT-MISSING") is None


def test_sanitize_metadata_removes_full_text_like_fields_and_keeps_operational_fields():
    metadata = audit_log_store.sanitize_metadata(
        {
            "status": "PendingApproval",
            "approval_required": True,
            "blocked": False,
            "grounded_answer": "full answer",
            "workflow_response": {"grounded_answer": "full answer"},
            "citations": [{"text": "full evidence"}],
            "evidence_text": "full evidence",
        }
    )

    assert metadata == {
        "status": "PendingApproval",
        "approval_required": True,
        "blocked": False,
    }
