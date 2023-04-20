use ndarray::prelude::*;
use fft2d;
use rustfft::{num_complex::Complex, num_traits::Zero};

fn unflatten<T: Copy+Clone + Zero>(data: &Vec<T>, m: usize, n: usize) -> Array2<T> {
    let mut arr = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            arr[[i, j]] = data[i * n + j];
        }
    }
    arr
}

pub fn ifftshift<T: Copy+Clone + Zero+std::default::Default>(arr: & Array2<T>) -> Array2<T> {
    let (width, height) = arr.dim();
    let shifted = fft2d::slice::ifftshift(height, width, arr.as_slice().unwrap());
    let shifted_arr = unflatten(&shifted, width, height);
    shifted_arr
}

pub fn fftshift<T: Copy+Clone + Zero+std::default::Default>(data: &Array2<T>) -> Array2<T> {
    let (width, height) = data.dim();
    let shifted = fft2d::slice::fftshift(height, width, data.as_slice().unwrap());
    let shifted_arr = unflatten(&shifted, width, height);
    shifted_arr
}

pub fn ifft2(data: &Array2<Complex<f64>>) -> Array2<Complex<f64>>{
    // let mut temp = data.clone();
    let mut temp = transpose(data);
    let (width, height) = data.dim();
    let norm = (width * height) as f64;
    temp.mapv_inplace(|x| x / norm);
    fft2d::slice::ifft_2d(width, height, temp.as_slice_mut().unwrap());
    temp.into_shape((width, height)).unwrap()
    // temp
}

pub fn fftn(data: &Array2<Complex<f64>>) -> Array2<Complex<f64>>{
    let mut temp = transpose(data);
    let (width, height) = data.dim();
    fft2d::slice::fft_2d(width, height, temp.as_slice_mut().unwrap());
    temp.into_shape((width, height)).unwrap()

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

fn transpose<T: Copy+Clone + Zero+std::default::Default>(data: &Array2<T>) -> Array2<T> {
    let (width, height) = data.dim();
    let mut transposed = Array2::zeros((height, width));
    for ((i, j), value) in data.indexed_iter() {
        transposed[[j, i]] = *value;
    }
    transposed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fftshift_square() {
        // arr is enumared 4x4 arr
        let arr = array![
            [0., 1., 2., 3.],
            [4., 5., 6., 7.],
            [8., 9., 10., 11.],
            [12., 13., 14., 15.]
        ];
        // fftshifted is the same arr but shifted
        let fftshifted = array![
            [10., 11.,  8.,  9.],
            [14., 15., 12., 13.],
            [ 2.,  3.,  0.,  1.],
            [ 6.,  7.,  4.,  5.]
        ];
        let shifted = fftshift(&arr);
        assert_eq!(shifted, fftshifted);
    }

    #[test]
    fn test_fftshift_rectangle(){
        // arr is enumarated 4x6 arr
        let arr = array![
            [ 0.,  1.,  2.,  3.,  4.,  5.],
            [ 6.,  7.,  8.,  9., 10., 11.],
            [12., 13., 14., 15., 16., 17.],
            [18., 19., 20., 21., 22., 23.]
        ];
        let fftshifted = array![
            [15., 16., 17., 12., 13., 14.],
            [21., 22., 23., 18., 19., 20.],
            [ 3.,  4.,  5.,  0.,  1.,  2.],
            [ 9., 10., 11.,  6.,  7.,  8.]
        ];
        let shifted = fftshift(&arr);
        assert_eq!(shifted, fftshifted);
    }

    #[test]
    fn test_ifftshift_square() {
        // arr is enumared 4x4 arr
        let arr = array![
            [0., 1., 2., 3.],
            [4., 5., 6., 7.],
            [8., 9., 10., 11.],
            [12., 13., 14., 15.]
        ];
        // fftshifted is the same arr but shifted
        let fftshifted = array![
            [10., 11.,  8.,  9.],
            [14., 15., 12., 13.],
            [ 2.,  3.,  0.,  1.],
            [ 6.,  7.,  4.,  5.]
        ];
        let shifted = ifftshift(&fftshifted);
        assert_eq!(shifted, arr);
    }

    #[test]
    fn test_fftn_square(){
        let arr = array![
            [0., 1., 2., 3.],
            [4., 5., 6., 7.],
            [8., 9., 10., 11.],
            [12., 13., 14., 15.]
        ];
        let complex_arr = complexify(&arr);
        let fftn_arr = fftn(&complex_arr);
        let expected = array![
            [Complex::new(120.,0.),  Complex::new(-8.,8.),  Complex::new(-8.,0.),  Complex::new(-8.,-8.)],
            [Complex::new(-32.,32.),  Complex::new(0.,0.),  Complex::new(0.,0.),  Complex::new(0.,0.)],
            [Complex::new(-32.,0.),  Complex::new(0.,0.),  Complex::new(0.,0.),  Complex::new(0.,0.)],
            [Complex::new(-32.,-32.),  Complex::new(0.,0.),  Complex::new(0.,0.),  Complex::new(0.,0.)],
        ];
        assert_eq!(fftn_arr, expected);
    }

    #[test]
    fn test_fftn_rectangle(){
        let arr = array![
            [0., 1., 2., 3.],
            [4., 5., 6., 7.]
        ];
        let complex_arr = complexify(&arr);
        let fftn_arr = fftn(&complex_arr);
        let expected = array![
            [Complex::new(28.0,-0.0),Complex::new(-4.0,4.0),Complex::new(-4.0,-0.0),Complex::new(-4.0,-4.0)],
            [Complex::new(-16.0,-0.0),Complex::new(0.0,0.0),Complex::new(0.0,-0.0),Complex::new(0.0,-0.0)]
            ];
        assert_eq!(fftn_arr, expected);
        assert_eq!(fftn_arr.dim(), expected.dim());
        assert_eq!(fftn_arr[[1,1]], expected[[1,1]]);
        
    }

    #[test]
    fn test_ifft2_square(){
        let complex_arr = array![
            [Complex::new(120.,0.),  Complex::new(-8.,8.),  Complex::new(-8.,0.),  Complex::new(-8.,-8.)],
            [Complex::new(-32.,32.),  Complex::new(0.,0.),  Complex::new(0.,0.),  Complex::new(0.,0.)],
            [Complex::new(-32.,0.),  Complex::new(0.,0.),  Complex::new(0.,0.),  Complex::new(0.,0.)],
            [Complex::new(-32.,-32.),  Complex::new(0.,0.),  Complex::new(0.,0.),  Complex::new(0.,0.)],
        ];
        let iff2_arr = ifft2(&complex_arr);
        let arr = array![
            [0., 1., 2., 3.],
            [4., 5., 6., 7.],
            [8., 9., 10., 11.],
            [12., 13., 14., 15.]
        ];
        let expected = complexify(&arr);
        assert_eq!(iff2_arr, expected);
    }

    #[test]
    fn test_ifft2_rectangle(){
        let complex_arr = array![
            [Complex::new(28.0,-0.0),Complex::new(-4.0,4.0),Complex::new(-4.0,-0.0),Complex::new(-4.0,-4.0)],
            [Complex::new(-16.0,-0.0),Complex::new(0.0,0.0),Complex::new(0.0,-0.0),Complex::new(0.0,-0.0)]
        ];
        let iff2_arr = ifft2(&complex_arr);
        let arr: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = array![
            [0., 1., 2., 3.],
            [4., 5., 6., 7.]
        ];
        let expected = complexify(&arr);
        assert_eq!(iff2_arr, expected);
    }
}
