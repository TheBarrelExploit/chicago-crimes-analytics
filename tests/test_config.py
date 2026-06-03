def test_config_loads():
    """Verificar que se importe las configuraciones del proyecto"""
    from src.config import load_settings

    settings = load_settings()

    assert settings is not None
    assert settings.r2_bucket_name == "chicago-crimes-analytics"
    assert settings.api_port == 8000
