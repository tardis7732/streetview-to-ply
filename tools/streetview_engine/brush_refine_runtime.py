"""Bind installed Linux gsplat to the exact reference Python math and explicit library."""
from importlib import metadata,util
from pathlib import Path
import hashlib,sys
EXPECTED_PYTHON={'rendering.py':'6b5e4101035031afbb26bed7e496382c7502f9d8bc7cc9a432f290674a4d9133','cuda/_wrapper.py':'b710953ca19e47ac93583e67cb44ea4bb92b97d06ba67ae5d950a71c878dc4e4'}
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def renderer_provenance(library,expected_sha256):
 library=Path(library).resolve(strict=True)
 if sha(library)!=expected_sha256:raise ValueError('Declared renderer library changed')
 if metadata.version('gsplat')!='1.5.3':raise ValueError('L03 reference requires gsplat1.5.3')
 root=Path(util.find_spec('gsplat').origin).parent
 hashes={name:sha(root/name) for name in EXPECTED_PYTHON}
 if hashes!=EXPECTED_PYTHON:raise ValueError('gsplat Python renderer differs from recorded L03 source')
 return dict(gsplat_version='1.5.3',torch_version=metadata.version('torch'),python=sys.version,
  rendering_path=str(root/'rendering.py'),rendering_sha256=hashes['rendering.py'],wrapper_sha256=hashes['cuda/_wrapper.py'],
  extension_path=str(library),extension_sha256=expected_sha256,reference_python_byte_exact=True,
  platform_caveat='Same formulas; Linux compiled library and floating point execution need not be byte-identical to the historical Windows run.')
def assert_loaded_renderer(record):
 backend=sys.modules.get('gsplat.cuda._backend');extension=getattr(backend,'_C',None)
 if extension is None:raise ValueError('No loaded gsplat raster extension to verify')
 path=Path(extension.__file__).resolve(strict=True)
 if path!=Path(record['extension_path']) or sha(path)!=record['extension_sha256']:raise ValueError('Active gsplat extension differs from declared library')
 return str(path)