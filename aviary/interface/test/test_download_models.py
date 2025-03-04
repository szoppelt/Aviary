import subprocess
import unittest
from pathlib import Path
import shutil

from aviary.utils.functions import get_aviary_resource_path
from openmdao.utils.testing_utils import use_tempdirs
from aviary.interface.download_models import get_model, save_file


@use_tempdirs
class CommandEntryPointsTestCases(unittest.TestCase):

    def run_and_test_hangar(self, filenames, out_dir=''):
        # tests that the commands return an exit code of 0 and that the files are generated
        if isinstance(filenames, str):
            filenames = [filenames]
        cmd = ['aviary', 'hangar'] + filenames

        if out_dir:
            cmd += ['-o', out_dir]
            out_dir = Path(out_dir)
        else:
            out_dir = Path.cwd() / 'aviary_models'

        try:
            output = subprocess.check_output(cmd)
            for filename in filenames:
                path = out_dir / filename.split('/')[-1]
                self.assertTrue(path.exists())
        except subprocess.CalledProcessError as err:
            self.fail(f"Command '{cmd}' failed.  Return code: {err.returncode}")

    def test_single_file_without_path(self):
        filename = 'turbofan_22k.deck'
        self.run_and_test_hangar(filename)

    def test_single_file_with_path(self):
        filename = 'engines/turbofan_22k.deck'
        self.run_and_test_hangar(filename)

    def test_multiple_files(self):
        filenames = ['small_single_aisle_GASP.dat', 'small_single_aisle_GASP.csv']
        self.run_and_test_hangar(filenames)

    def test_folder(self):
        filename = 'engines'
        self.run_and_test_hangar(filename)

    def test_single_file_custom_outdir(self):
        filename = 'small_single_aisle_GASP.csv'
        out_dir = '~/test_hangar'
        self.run_and_test_hangar(filename, out_dir)
        shutil.rmtree(out_dir)

    def test_expected_path(self):
        aviary_path = get_model('large_single_aisle_1_GASP.dat')

        expected_path = get_aviary_resource_path(
            'models/large_single_aisle_1/large_single_aisle_1_GASP.dat'
        )
        self.assertTrue(str(aviary_path) == str(expected_path))


if __name__ == "__main__":
    unittest.main()
