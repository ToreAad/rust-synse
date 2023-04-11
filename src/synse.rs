use ndarray::Array2;
use rustfft::num_complex::Complex;
use crate::fft::{ifft2, fftn, complexify, ifftshift,ifftshift_c, fftshift, realify};
use crate::psf::{get_psf};


pub fn get_grad(m: usize, n: usize) -> Array2<Complex<f64>> {
    let mut grad_kernel =  Array2::zeros((m, n));
    grad_kernel[[m / 2, 1+ n / 2 ]] = 1.0;
    grad_kernel[[m / 2,  n / 2]] = -1.0;
    let mut arr = spat_to_freq(&grad_kernel);
    ifftshift_c(&mut arr);
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
    let psf = get_psf(m as i32, n as i32, a1, a2, l1, l2, s); 
    let complex_psf = complexify(&psf);
    // sfft.Center2(complex_psf)
    complex_psf
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
    freq_to_spat(&masked_f)
}

pub fn freq_to_spat(data: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let mut temp = data.clone();
    ifft2(&mut temp);
    temp
}

pub fn spat_to_freq(data: &Array2<f64>) -> Array2<Complex<f64>> {
    let mut complex_data = complexify(data);
    fftn(&mut complex_data);
    complex_data
}

pub fn make_seismic_from_strata(strata: &Array2<f64>, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<f64> {
    let mut convolved = do_convolve(strata, a1, a2, l1, l2, s);
    fftshift(&mut convolved);
    realify(&convolved)
}

pub fn get_psf_frequency_domain(m: usize, n: usize, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<f64> {
    let psf_freq = get_psf(m as i32, n as i32, a1, a2, l1, l2, s); 
    psf_freq
}

pub fn get_psf_spatial_domain(m: usize, n: usize, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<f64> {
    let psf_freq = get_psf(m as i32, n as i32, a1, a2, l1, l2, s); 
    let psf_freq_complex = complexify(&psf_freq);
    let mut psf_spat = freq_to_spat(&psf_freq_complex);
    fftshift(&mut psf_spat);
    realify(&psf_spat)
}

#[cfg(test)]
mod tests {
	use super::*;

    #[test]
    fn test_get_grad() {
        let grad = get_grad(5, 5);
        assert_eq!(grad[[2, 2]], Complex::new(0.0, 0.0));
        assert_eq!(grad[[2, 3]], Complex::new(1.0, 0.0));
        assert_eq!(grad[[2, 1]], Complex::new(-1.0, 0.0));
    }
}