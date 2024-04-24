from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

setup(name='custom_api_cpp',
      ext_modules=[cpp_extension.CppExtension(
                                          name='custom_api_cpp',
                                          sources=['custom_api.cpp'],
                                          extra_compile_args=['-fopenmp', '-O3','-std=c++17', '-I%s/tbb/include' %os.environ['PATH_LAZYDP']],
                                          extra_link_args=['-Wl,-rpath,%s/tbb/build/linux_intel64_gcc_cc9.4.0_libc2.27_kernel4.15.0_release' %os.environ['PATH_LAZYDP']],
                                          library_dirs=['%s/tbb/build/linux_intel64_gcc_cc9.4.0_libc2.27_kernel4.15.0_release' %os.environ['PATH_LAZYDP']],
                                          libraries=['tbb']
                  )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
