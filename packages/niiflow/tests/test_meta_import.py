def test_import_meta_and_subpackages() -> None:
    import niiflow  # noqa: F401
    import niiflow.core  # noqa: F401
    import niiflow.preproc  # noqa: F401
