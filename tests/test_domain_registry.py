from domains import DOMAIN_REGISTRY, get_adapter


def test_registry_has_default_domain():
    assert "msme_startup" in DOMAIN_REGISTRY


def test_get_adapter_returns_expected_adapter():
    adapter = get_adapter("msme_startup")
    assert adapter.domain_id == "msme_startup"
    assert adapter.time_horizon == 36
    assert adapter.total_entities == 30
    assert "wait_and_observe" in adapter.valid_actions

