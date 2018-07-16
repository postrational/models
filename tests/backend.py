from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import glob
import os
import re
import shutil
import tarfile
import tempfile
from os.path import dirname

from six.moves.urllib.request import urlretrieve

import onnx.backend.test
from onnx.backend.test.case.test_case import TestCase as OnnxTestCase


class ModelTestRunner(onnx.backend.test.BackendTest):

    def __init__(self, backend, parent_module=None):  # type: (Type[Backend], Optional[str]) -> None
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns = set()  # type: Set[Pattern[Text]]
        self._exclude_patterns = set()  # type: Set[Pattern[Text]]
        self._test_items = defaultdict(dict)  # type: Dict[Text, Dict[Text, TestItem]]

        testcases = []
        model_url_regex = r'(http.+\/opset_(.+)\/(.+)\.tar\.gz)'
        model_zoo_root = dirname(dirname(__file__))
        model_readme_files = glob.glob(model_zoo_root + '/**/README.md')

        for model_readme_filename in model_readme_files:
            with open(model_readme_filename, 'r') as readme:
                for line in readme:
                    match = re.search(model_url_regex, line)
                    if match:
                        url, opset_version, model_name = match.groups()
                        model_name = "{}_opset{}".format(model_name, opset_version)
                        test_name = "test_{}".format(model_name)

                        test_case = OnnxTestCase(
                                name=test_name,
                                url=url,
                                model_name=model_name,
                                model_dir=None,
                                model=None,
                                data_sets=None,
                                kind='OnnxBackendRealModelTest',
                            )
                        testcases.append(test_case)

        for test_case in testcases:
            self._add_model_test(test_case, 'Zoo')

    def _prepare_model_data(self, model_test):  # type: (TestCase) -> Text
        onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', os.path.join('~', '.onnx')))
        models_dir = os.getenv('ONNX_MODELS',
                               os.path.join(onnx_home, 'models'))
        model_dir = os.path.join(models_dir, model_test.model_name)  # type: Text

        if not os.path.exists(os.path.join(model_dir, 'model.onnx')):
            if os.path.exists(model_dir):
                bi = 0
                while True:
                    dest = '{}.old.{}'.format(model_dir, bi)
                    if os.path.exists(dest):
                        bi += 1
                        continue
                    shutil.move(model_dir, dest)
                    break

            # On Windows, NamedTemporaryFile can not be opened for a
            # second time
            download_file = tempfile.NamedTemporaryFile(delete=False)
            try:
                download_file.close()
                print('Start downloading model {} from {}'.format(
                    model_test.model_name, model_test.url))
                urlretrieve(model_test.url, download_file.name)
                print('Done')
                with tarfile.open(download_file.name) as t:
                    top_level_directory_in_tar = t.getmembers()[0].name
                    t.extractall(models_dir)
                    extracted_dir = os.path.join(models_dir, top_level_directory_in_tar)
                    shutil.move(extracted_dir, model_dir)
            except Exception as e:
                print('Failed to prepare data for model {}: {}'.format(
                    model_test.model_name, e))
                raise
            finally:
                os.remove(download_file.name)
        return model_dir
