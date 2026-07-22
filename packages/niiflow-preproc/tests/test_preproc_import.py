def test_import_preproc() -> None:
    import niiflow.preproc as preproc

    assert preproc.DynamicPreprocessingWorkflow is not None
    assert preproc.RunPlan is not None
    assert preproc.dynamic_pipeline is not None
    assert preproc.dynamic_workflow is not None
    assert preproc.load_config is not None


def test_import_image_wrappers() -> None:
    from niiflow.preproc.functional.image import (
        ants_apply_transforms,
        ants_bias_field_correction,
        ants_brain_extraction,
        ants_denoise,
        ants_apply_mask,
        ants_registration,
        ants_reorient,
        ants_resample,
        ants_resample_to_target,
        ants_preprocess_brain_image,
    )

    assert callable(ants_apply_transforms)
    assert callable(ants_bias_field_correction)
    assert callable(ants_brain_extraction)
    assert callable(ants_denoise)
    assert callable(ants_apply_mask)
    assert callable(ants_registration)
    assert callable(ants_reorient)
    assert callable(ants_resample)
    assert callable(ants_resample_to_target)
    assert callable(ants_preprocess_brain_image)
