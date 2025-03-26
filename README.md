<p align="center">
  <img src="https://github.com/user-attachments/assets/11bc8f07-60cb-4628-bc88-5dee4b22bdc6" width="200" height="200">
</p>

<h1 align="center">Rubix NumPower</h1>
<p align="center">
<img width="500" height="100" src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=2500&pause=250&color=F7381D&multiline=true&width=500&height=100&separator=%3C&lines=use+NumPower;%3C%24arr+%3D+NumPower%3A%3Aarray(%5B%5B1%2C+2%5D%2C+%5B3%2C+4%5D%5D);%3C%24result+%3D+%24arr+*+2;%3Cecho+%24result;" alt="Typing code" />
</p>

Inspired by NumPy, the NumPower extension was created by Henrique Borba to provide the foundation for efficient scientific computing in PHP, as well as leverage the machine learning tools and libraries that already exist and can benefit from it.

This C extension developed for PHP can be used to considerably speed up mathematical operations on large datasets and facilitate the manipulation, creation and operation of N-dimensional tensors.

NumPower was designed from the ground up to utilize AVX2 and the GPU to further improve performance. With the use of contiguous single precision arrays, slices, buffer sharing and a specific GC engine, 

NumPower aims to manage memory more efficiently than a matrix in PHP arrays

<p align="center">
<img width="500" height="125" src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=2500&pause=250&color=F7381D&multiline=true&width=500&height=125&separator=%3C&lines=use+NumPower;%3C%24a+%3D+NumPower%3A%3Anormal(%5B2%2C+2%5D)-%3Egpu();%3C%24b+%3D+NumPower%3A%3Anormal(%5B2%2C+2%5D)-%3Egpu();%3C%24result+%3D+NumPower%3A%3Amatmul(%24aGpu%2C+%24bGpu);%3Cecho+%24result;" alt="Typing code" />
</p>

## Requirements
- PHP 8.x
- LAPACKE
- OpenBLAS
- **Optional**: Intel MKL
- **Optional (GPU)**: CUBLAS, CUDA Build Toolkit and cuDNN
- **Optional (Image)**: PHP-GD

## Compiling

``` 
$ phpize
$ ./configure
$ make install
```

## Compiling with GPU (CUDA) support

``` 
$ phpize
$ ./configure --with-cuda
$ make install-cuda
```

## GPU support

If you have an NVIDIA graphics card with CUDA support, you can use your graphics card 
to perform operations. To do this, just copy your array to the GPU memory.

```php
use \NumPower;

$x = NumPower::ones([10, 10]);
$y = NumPower::ones([10, 10]);

$xGpu = $x->gpu();   // Copy $x from RAM to VRAM
$yGpu = $y->gpu();   // Copy $y from RAM to VRAM

$r = NumPower::matmul($xGpu, $yGpu); // Matmul is performed using CUDA
```

Both GPU and CPU memory management are done automatically by NumPower, so the memory of both devices will be 
automatically freed by the garbage collector.  You can also bring arrays back from VRAM into RAM:

```php 
$xCpu = $x->cpu();
```

> **You must explicitly copy the arrays you want to use in your devices**. Cross-array operations (like adding) will 
> raise an exception if the arrays used are on different devices.
