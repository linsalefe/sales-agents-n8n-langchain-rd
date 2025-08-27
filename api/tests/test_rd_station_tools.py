# api/tests/test_rd_station_tools.py
import json
import respx
import httpx
import pytest
from agents.tools.rd_station_tools import RDStationClient, RDStationError

BASE_URL = "https://api.rd.services"
TOKEN = "test-token"


@pytest.fixture
def client():
    c = RDStationClient(base_url=BASE_URL, access_token=TOKEN, timeout=2.0)
    yield c
    c.close()


@respx.mock
def test_add_tags_success(client):
    route = respx.post(f"{BASE_URL}/platform/contacts/tags").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    out = client.add_tags("maria@email.com", ["alto-fit"])
    assert route.called
    assert out == {"ok": True}


@respx.mock
def test_update_stage_success(client):
    deal_id = "D123"
    route = respx.put(f"{BASE_URL}/crm/deals/{deal_id}/stage").mock(
        return_value=httpx.Response(200, json={"stage": "Qualificação"})
    )
    out = client.update_stage(deal_id, "Qualificação")
    assert route.called
    assert out["stage"] == "Qualificação"


@respx.mock
def test_create_task_success(client):
    deal_id = "D123"
    route = respx.post(f"{BASE_URL}/crm/deals/{deal_id}/tasks").mock(
        return_value=httpx.Response(200, json={"task_id": "T1"})
    )
    out = client.create_task(deal_id, "Ligar", "2025-08-28", owner="SDR-Ana")
    assert route.called
    assert out["task_id"] == "T1"


@respx.mock
def test_add_note_4xx_raises(client):
    deal_id = "D123"
    route = respx.post(f"{BASE_URL}/crm/deals/{deal_id}/notes").mock(
        return_value=httpx.Response(400, json={"error": "invalid"})
    )
    with pytest.raises(RDStationError):
        client.add_note(deal_id, "nota inválida")
    assert route.called


@respx.mock
def test_send_email_timeout_retry(client, monkeypatch):
    # Primeiro request: timeout; Segundo: sucesso
    route = respx.post(f"{BASE_URL}/marketing/emails/send")
    route.side_effect = [
        httpx.ReadTimeout("timeout"),
        httpx.Response(200, json={"sent": True}),
    ]
    out = client.send_email("to@email.com", "Subj", "Body", sender="SDR")
    assert out["sent"] is True
    assert route.call_count == 2
