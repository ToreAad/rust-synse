use ndarray::Array2;
use rustfft::{num_complex::Complex};
use fft2d;
use crate::{psf::{get_psf}, fft::{ifftshift, ifft2, fftn, fftshift}};

pub fn get_grad(m: usize, n: usize) -> Array2<Complex<f64>> {
    let mut grad_kernel =  Array2::zeros((m, n));
    grad_kernel[[m / 2, 1+ n / 2 ]] = 1.0;
    grad_kernel[[m / 2,  n / 2]] = -1.0;
    let mut arr = spat_to_freq(&grad_kernel);
    // ifftshift(&arr)
    arr
}

pub fn get_convolver(m: usize, n: usize, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<Complex<f64>> {
    let grad_mask = get_grad(m, n);
    let psf_mask = get_psf_freq(m, n, a1, a2, l1, l2, s);
    let mut convolved = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            convolved[[i, j]] = psf_mask[[i, j]] * grad_mask[[i, j]];
        }
    }
    convolved
}

pub fn get_psf_freq(m: usize, n: usize, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<Complex<f64>> {
    let mut psf = get_psf(m as i32, n as i32, a1, a2, l1, l2, s); 
    let shifted = ifftshift(&mut psf);
    complexify(&shifted)
}

pub fn get_psf_spat(m: usize, n: usize, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<Complex<f64>> {
    let psf_freq = get_psf_freq(m, n, a1, a2, l1, l2, s);
    let psf_spat = freq_to_spat(&psf_freq);
    psf_spat
}

pub fn do_convolve(data: &Array2<f64>, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<Complex<f64>> {
    let (m, n) = data.dim();
    let convolver = get_convolver(m, n, a1, a2, l1, l2, s);
    let f = spat_to_freq(&data);
    let mut masked_f = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            masked_f[[i, j]] = f[[i, j]] * convolver[[i, j]];
        }
    }
    let seis = freq_to_spat(&masked_f);
    fftshift(&seis)
}

fn complexify(data: &Array2<f64>) -> Array2<Complex<f64>> {
    let mut complex_data = Array2::zeros(data.dim());
    for ((i, j), value) in data.indexed_iter() {
        complex_data[[i, j]] = Complex::new(*value, 0.0);
    }
    complex_data
}

fn realify(data: &Array2<Complex<f64>>) -> Array2<f64> {
    let mut complex_data = Array2::zeros(data.dim());
    for ((i, j), value) in data.indexed_iter() {
        complex_data[[i, j]] = value.re;
    }
    complex_data
}

pub fn freq_to_spat(data: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    ifft2(&data)
}

pub fn spat_to_freq(data: &Array2<f64>) -> Array2<Complex<f64>> {
    let data = complexify(&data);
    fftn(&data)
}


pub fn make_seismic_from_strata(strata: &Array2<f64>, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<f64> {
    let mut convolved = do_convolve(strata, a1, a2, l1, l2, s);
    let shifted = fftshift(&convolved);
    realify(&shifted)
}

pub fn get_psf_frequency_domain(m: usize, n: usize, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<f64> {
    let psf_freq: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = get_psf(m as i32, n as i32, a1, a2, l1, l2, s); 
    psf_freq
}

pub fn get_psf_spatial_domain(m: usize, n: usize, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<f64> {
    let psf_freq = get_psf(m as i32, n as i32, a1, a2, l1, l2, s);
    let psf_freq_complex = complexify(&psf_freq);
    let psf_spat = freq_to_spat(&psf_freq_complex);
    let shifted = fftshift(&psf_spat);
    realify(&shifted)
}



#[cfg(test)]
mod tests {
	use std::fs::File;

    use rustfft::num_traits::Float;

    use crate::converters::{get_strata_from_gray_scale_image, write_seismic_png_image};

    use super::*;

    #[test]
    fn test_get_grad() {
        let grad = get_grad(4, 4);
        assert_eq!(grad[[2,2]], Complex::new(-2.0, 0.0));
        assert_eq!(grad[[3,2]], Complex::new(2.0, 0.0));
        assert_eq!(grad[[1,2]], Complex::new(2.0, 0.0));
        assert_eq!(grad[[2,3]], Complex::new(1.0, -1.0));
        assert_eq!(grad[[2,1]], Complex::new(1.0, 1.0));
    }

    #[test]
    fn test_write_grad_to_disk_as_img(){
        let grad = get_grad(40, 80);
        let mut imgbuf = image::ImageBuffer::new(40, 80);
        let mx = grad.map(|x| x.re).fold(0.0, |a, b| a.max(*b));
        let mn = grad.map(|x| x.re).fold(0.0, |a, b| a.min(*b));
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            let val = grad[[x as usize, y as usize]].re;
            let val = ((val - mn) / (mx - mn) * 255.0) as u8;
            *pixel = image::Rgb([val, val, val]);
        }
        imgbuf.save("grad.png").unwrap();
    }

    #[test]
    fn test_write_convolver_to_disk_as_img(){
        let convolver = get_convolver(40, 80, 45.0, 180.0-45.0, 0.75, 0.75, 0.010);
        let mut imgbuf = image::ImageBuffer::new(40, 80);
        let mx = convolver.map(|x| x.re).fold(0.0, |a, b| a.max(*b));
        let mn = convolver.map(|x| x.re).fold(0.0, |a, b| a.min(*b));
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            let val = convolver[[x as usize, y as usize]].re;
            let val = (val - mn) / (mx - mn);
            let val = (val * 255.0) as u8;
            *pixel = image::Luma([val]);
        }
        imgbuf.save("convolver.png").unwrap();
    }
    

    #[test]
    fn test_write_psf_freq_to_disk_as_img(){
        let psf_freq = get_psf_freq(40, 80, 45.0, 180.0-45.0, 0.75, 0.75, 0.010);
        let mut imgbuf = image::ImageBuffer::new(40, 80);
        let mx = psf_freq.map(|x| x.re).fold(0.0, |a, b| a.max(*b));
        let mn = psf_freq.map(|x| x.re).fold(0.0, |a, b| a.min(*b));
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            let val = psf_freq[[x as usize, y as usize]].re;
            let val = (val - mn) / (mx - mn);
            let val = (val * 255.0) as u8;
            *pixel = image::Luma([val]);
        }
        imgbuf.save("psf_freq.png").unwrap();
    }

    #[test]
    fn test_write_psf_spat_to_disk_as_img(){
        let psf_freq = get_psf_spat(40, 80, 45.0, 180.0-45.0, 0.75, 0.75, 0.010);
        let mut imgbuf = image::ImageBuffer::new(40, 80);
        let mx = psf_freq.map(|x| x.re).fold(0.0, |a, b| a.max(*b));
        let mn = psf_freq.map(|x| x.re).fold(0.0, |a, b| a.min(*b));
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            let val = psf_freq[[x as usize, y as usize]].re;
            let val = (val - mn) / (mx - mn);
            let val = (val * 255.0) as u8;
            *pixel = image::Luma([val]);
        }
        imgbuf.save("psf_spat.png").unwrap();
    }

    #[test]
    fn test_write_synthetic_seismic_to_disk_as_img(){
        let strata = get_strata_from_gray_scale_image("strata.png");
        let convolved = do_convolve(&strata, 45.0, 180.0-45.0, 0.75, 0.75, 0.010);
        let convolved_real = realify(&convolved);
        let mut file = File::create("convolved.png").expect("Failed to create file");
        write_seismic_png_image(&convolved_real, &mut file).expect("Failed to write image");
    }



}