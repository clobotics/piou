from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='pixel_weights_cpp',
    ext_modules=[
        CppExtension('pixel_weights_cpu', [
            'pixel_weights_binding.cpp'
        ]),
        CUDAExtension('pixel_weights_cuda', [
            'pixel_weights_cuda.cpp',
            'pixel_weights_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
