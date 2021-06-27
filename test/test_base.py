import os
from unittest import TestCase
import matplotlib.pyplot as plt

class TestBase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.AE_DATA_PATH = os.environ['AE_DATA'].strip('"') if 'AE_DATA' in os.environ else None
        cls.AE_TEST_PATH = os.environ['AE_TEST'].strip('"') if 'AE_TEST' in os.environ else None
        # cls.kurucz_grid = None



    # def get_kurucz_grid(self):
    #     raise NotImplementedError()

    #     if self.kurucz_grid is None:
    #         file = os.path.join(self.AE_DATA_PATH, 'stellar/compressed/kurucz.h5')
    #         self.kurucz_grid = KuruczGrid(model='test')
    #         self.kurucz_grid.load(file, s=None, format='h5')

    #     return self.kurucz_grid


    # def setUp(self):
    #     plt.figure(figsize=(10, 6))

    # def get_filename(self, ext):
    #     filename = type(self).__name__[4:] + '_' + self._testMethodName[5:] + ext
    #     return filename


    # def save_fig(self, f=None, filename=None):
    #     if f is None:
    #         f = plt
    #     if filename is None:
    #         filename = self.get_filename('.png')
    #     f.savefig(os.path.join(self.AE_TEST_PATH, filename))