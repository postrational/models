from ngraph_onnx.onnx_importer.backend import NgraphBackend
from .backend import ModelTestRunner

# Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
NgraphBackend.backend_name = 'CPU'

# import all test cases at global scope to make them visible to python.unittest
backend_test = ModelTestRunner(NgraphBackend, __name__)

globals().update(backend_test.enable_report().test_cases)
