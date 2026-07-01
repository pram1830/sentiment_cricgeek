import builtins

from eqs_service import EQSService


def test_local_pipeline_fallback_is_deterministic(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentiment_engine.sentiment_pipeline":
            raise ImportError("simulated missing pipeline")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = EQSService._score_with_local_pipeline(
        "India won the final over with a strong finish and smart bowling plans."
    )

    assert result["success"] is True
    assert 20.0 <= result["score"] <= 95.0
    assert "components" in result
