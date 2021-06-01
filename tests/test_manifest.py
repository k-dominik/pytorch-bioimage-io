from pathlib import Path

import pytest
import yaml

from bioimageio.spec import utils, load_and_resolve_spec

MANIFEST_PATH = Path(__file__).parent.parent / "manifest.yaml"


def pytest_generate_tests(metafunc):
    if "category" in metafunc.fixturenames and "spec_path" in metafunc.fixturenames:
        with MANIFEST_PATH.open() as f:
            manifest = yaml.safe_load(f)

        categories_and_spec_paths = [
            (category, spec_path) for category, spec_paths in manifest.items() for spec_path in spec_paths
        ]
        metafunc.parametrize("category,spec_path", categories_and_spec_paths)


def test_load_specs_from_manifest(category, spec_path):

    spec_path = MANIFEST_PATH.parent / spec_path
    assert spec_path.exists()

    loaded_spec = load_and_resolve_spec(spec_path)
    instance = utils.get_instance(loaded_spec)
    assert instance
