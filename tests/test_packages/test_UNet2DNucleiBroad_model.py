from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
from zipfile import ZipFile

from bioimageio.spec import load_and_resolve_spec
from bioimageio.spec.utils import get_instance

MODEL_EXTENSIONS = (".model.yaml", ".model.yml")
UNET_2D_NUCLEI_BROAD_PACKAGE_URL = (
    "https://github.com/bioimage-io/pytorch-bioimage-io/releases/download/v0.1.1/UNet2DNucleiBroad.model.zip"
)


def guess_model_path(file_names: List[str]) -> Optional[str]:
    for file_name in file_names:
        if file_name.endswith(MODEL_EXTENSIONS):
            return file_name

    return None


def eval_model_zip(model_zip: ZipFile):
    with TemporaryDirectory() as tempdir:
        temp_path = Path(tempdir)
        model_zip.extractall(temp_path)
        spec_file_str = guess_model_path([str(file_name) for file_name in temp_path.glob("*")])
        bioimageio_model = load_and_resolve_spec(spec_file_str)

        return get_instance(bioimageio_model)
