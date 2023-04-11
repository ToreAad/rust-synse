use ndarray::prelude::*;
use fft2d;
use rustfft::num_complex::Complex;


pub fn ifftshift(data: &mut Array2<f64>) {
    let (width, height) = data.dim();
    fft2d::slice::ifftshift(width, height, data.as_slice_mut().unwrap());
}

pub fn ifftshift_c(data: &mut Array2<Complex<f64>>) {
    let (width, height) = data.dim();
    fft2d::slice::ifftshift(width, height, data.as_slice_mut().unwrap());
}

pub fn fftshift(data: &mut Array2<Complex<f64>>) {
    let (width, height) = data.dim();
    fft2d::slice::fftshift(width, height, data.as_slice_mut().unwrap());
}

pub fn ifft2(data: &mut Array2<Complex<f64>>) {
    let (width, height) = data.dim();
    fft2d::slice::ifft_2d(width, height, data.as_slice_mut().unwrap());
}

pub fn fftn(data: &mut Array2<Complex<f64>>) {
    let (width, height) = data.dim();
    fft2d::slice::fft_2d(width, height, data.as_slice_mut().unwrap());
}

pub fn complexify(data: &Array2<f64>) -> Array2<Complex<f64>> {
    let mut complex_data = Array2::zeros(data.dim());
    for ((i, j), value) in data.indexed_iter() {
        complex_data[[i, j]] = Complex::new(*value, 0.0);
    }
    complex_data
}

pub fn realify(data: &Array2<Complex<f64>>) -> Array2<f64> {
    let mut complex_data = Array2::zeros(data.dim());
    for ((i, j), value) in data.indexed_iter() {
        complex_data[[i, j]] = value.re;
    }
    complex_data
}
