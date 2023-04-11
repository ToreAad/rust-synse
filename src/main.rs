use psf::get_psf;

mod fft;
mod psf;
mod synse;

fn main() {
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
