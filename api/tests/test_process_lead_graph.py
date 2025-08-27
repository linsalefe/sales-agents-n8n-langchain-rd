# api/tests/test_process_lead_graph.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_process_lead_graph_e2e():
    payload = {
        "lead": {
            "name": "Maria Silva",
            "email": "maria.silva@email.com",
            "phone": "+55 11 98765-4321",
            "profession": "Psicóloga",
            "source": "Indicação de colega",
            "utm_source": "facebook",
            "utm_medium": "cpc",
            "utm_campaign": "pos_saude_mental",
        },
        "context": {
            "product": "Pós-graduação em Saúde Mental",
            "region": "BR",
            "language": "pt-BR",
            "icp_profile": {
                "segment": "Educação / Saúde",
                "company_size": "1-50",
                "role": "Coordenadora",
            },
        },
    }

    r = client.post("/process-lead", json=payload)
    assert r.status_code == 200
    data = r.json()

    # chaves principais
    for key in ["status", "lead_score", "reasoning", "crm_actions", "messaging"]:
        assert key in data

    # regras básicas do grafo
    assert data["status"] in {"qualificado", "avaliar"}
    assert isinstance(data["lead_score"], int)
    assert 0 <= data["lead_score"] <= 100

    # crm_actions
    crm = data["crm_actions"]
    assert crm["stage"] in {"Qualificação", "Novo Lead"}
    assert isinstance(crm["tags"], list)
    assert isinstance(crm["tasks"], list)
    assert isinstance(crm["note"], str)
    assert isinstance(crm["next_step_date"], str)

    # messaging
    msg = data["messaging"]
    assert "email" in msg and "whatsapp" in msg and "call_script" in msg
    assert "subject" in msg["email"] and "body" in msg["email"]

    # cenário esperado de alto fit (do payload acima)
    assert data["status"] == "qualificado"
    assert data["lead_score"] >= 70
    assert "alto-fit" in crm["tags"]
    assert "pos-graduacao-em-saude-mental" in crm["tags"]
