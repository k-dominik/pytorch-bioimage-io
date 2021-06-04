from bioimageio.spec.__main__ import verify_bioimageio_manifest


def test_validate_manifest(manifest_path):
    verify_bioimageio_manifest(manifest_path)
