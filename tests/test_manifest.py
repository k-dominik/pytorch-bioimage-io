from ruamel.yaml import YAML

from bioimageio.spec.__main__ import verify_bioimageio_manifest_data

yaml = YAML(typ="safe")


def test_validate_manifest(manifest_path):
    verify_bioimageio_manifest_data(yaml.load(manifest_path), auto_convert=True)
