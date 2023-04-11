use ndarray::Array2;

fn gaussian(x: f64, mu: f64, sig: f64) -> f64 {
    (-(x-mu).powi(2) / (2.0 * sig.powi(2))).exp()
}

fn get_gaussian(x: f64, dev: f64) -> f64 {
    let mn = gaussian(0.0, 0.5, dev);
    let val = gaussian(x, 0.5, dev);
    let mx = 1.0;
    let result = (val - mn) / (mx - mn);
    return result;
}

fn r_eclipse(a: f64, b: f64, o: f64) -> f64 {
	a * b / ((b * o.cos()).powi(2) + (a * o.sin()).powi(2)).powf(0.5)
}

fn transform_angle(l: f64, o: f64) -> f64 {
	let y1 = o.sin();
	let x1 = o.cos();
	(l*y1).atan2(x1)
}

pub fn get_psf(m: i32, n: i32, a1: f64, a2: f64, l1: f64, l2: f64, s: f64) -> Array2<f64> {
	let mut mask = Array2::zeros((m as usize, n as usize));
	// let mut a1 = a1 - 90.0;
	// let mut a2 = a2 - 90.0;
	for x in -(m / 2)..(m / 2) {
		for y in -(n / 2)..(n / 2) {
			let i_x = x + m / 2;
			let i_y = y + n / 2;
			let mxl1 = (l1 * (m as f64) * 0.5).min((m as f64) * 0.5);
			let mxl2 = (l2 * mxl1).min( (n as f64) * 0.5);
			let fx = x as f64;
			let fy = y as f64;
			let r = (fx * fx + fy * fy).sqrt();
			let o = -(fx.atan2(fy));
			let max_len = r_eclipse(mxl1, mxl2, o);
			let o1 = transform_angle(l2, a1 * std::f64::consts::PI / 180.0);
			let o2 = transform_angle(l2, a2 * std::f64::consts::PI / 180.0);
			if (r < max_len) && (o > o1) && (o < o2) {
				let dev = 0.25 * (1.0 - s);
				let val = get_gaussian(r / max_len, dev);
				// TODO: Somethign is wrong here
				if s == 0.0 || (val).abs() > 1e-1 {
					mask[[i_x as usize, (n - i_y) as usize]] = val;
				} 
			}
		}
	}
	mask
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_get_psf() {
		let psf = get_psf(300, 400, 45.0, 180.0-45.0, 0.75, 0.75, 0.010);
		// write to grayscale image file
		let mut imgbuf = image::ImageBuffer::new(psf.shape()[1] as u32, psf.shape()[0] as u32);
		for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
			let val = psf[[y as usize, x as usize]];
			let val = (val * 255.0).min(255.0).max(0.0) as u8;
			*pixel = image::Luma([val]);
		}
		imgbuf.save("psf.png").unwrap();
	}
}