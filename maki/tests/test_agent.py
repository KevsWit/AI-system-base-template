from .. import Maki

def test_basic_routes_and_trace():
    mk = Maki()

    # GYM
    r1 = mk.ask("¿Qué planes de gimnasio tienen?")
    assert isinstance(r1, dict)
    assert "final_answer" in r1 and isinstance(r1["final_answer"], str)
    assert "trace" in r1 and isinstance(r1["trace"], list)
    assert r1["domain"] in ("GYM", "CLIENTS", "UNKNOWN")
    assert r1["domain"] == "GYM", f"Dominio esperado GYM, obtenido {r1['domain']}"
    assert any(tok in r1["final_answer"].lower() for tok in ["plan", "precio", "mensual", "trimestral", "anual"])

    # CLIENTS
    r2 = mk.ask("¿Qué canal de contacto prefieren los clientes?")
    assert isinstance(r2, dict)
    assert "final_answer" in r2 and isinstance(r2["final_answer"], str)
    assert "trace" in r2 and isinstance(r2["trace"], list)
    assert r2["domain"] in ("GYM", "CLIENTS", "UNKNOWN")
    assert r2["domain"] == "CLIENTS", f"Dominio esperado CLIENTS, obtenido {r2['domain']}"
    assert any(tok in r2["final_answer"] for tok in ["WhatsApp", "whatsapp", "contacto"])

    # UNKNOWN -> debe producir respuesta (fallback si aplica) y tener trace
    r3 = mk.ask("Cuéntame sobre astrofísica cuántica del corazón de las estrellas")
    assert isinstance(r3, dict)
    assert "final_answer" in r3 and isinstance(r3["final_answer"], str)
    assert "trace" in r3 and isinstance(r3["trace"], list)
    assert r3["domain"] in ("GYM", "CLIENTS", "UNKNOWN")

    # El trace debe registrar welcome, retrieve, route, research, verify
    step_names = [t[0] for t in r1["trace"]]
    needed = {"tool.welcome", "retrieve.counts", "route.label", "route.domain", "research.draft", "verify.report"}
    assert needed.issubset(set(step_names)), f"Trace incompleto, faltan: {needed - set(step_names)}"
