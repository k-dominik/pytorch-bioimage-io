from pathlib import Path

import yaml

from bioimageio.spec import load_and_resolve_spec, utils
from bioimageio.spec.__main__ import verify_bioimageio_manifest


def pytest_generate_tests(metafunc):
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    if "category" in metafunc.fixturenames and "spec_path" in metafunc.fixturenames:
        with manifest_path.open() as f:
            manifest = yaml.safe_load(f)

        categories_and_spec_paths = [
            (category, manifest_path.parent / spec_path) for category, spec_paths in manifest.items() for spec_path in spec_paths
        ]
        metafunc.parametrize("category,spec_path", categories_and_spec_paths)


def test_validate_manifest(manifest_path):
    verify_bioimageio_manifest(manifest_path)


def test_load_specs_from_manifest(category, spec_path):
    assert spec_path.exists()

    loaded_spec = load_and_resolve_spec(spec_path)
    instance = utils.get_instance(loaded_spec)
    assert instance
