def test_import_preproc() -> None:
    import niiflow.preproc  # noqa: F401


def test_import_image_wrappers() -> None:
    from niiflow.preproc.functional.image import (
        apply_transforms_ants,
        bias_field_correction_ants,
        brain_extraction_ants,
        denoise_ants,
        mask_image_ants,
        preprocessing_pipeline_ants,
        registration_ants,
        resample_ants,
        resample_to_target_ants,
    )

    assert callable(apply_transforms_ants)
    assert callable(bias_field_correction_ants)
    assert callable(brain_extraction_ants)
    assert callable(denoise_ants)
    assert callable(mask_image_ants)
    assert callable(preprocessing_pipeline_ants)
    assert callable(registration_ants)
    assert callable(resample_ants)
    assert callable(resample_to_target_ants)
