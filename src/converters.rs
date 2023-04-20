// Use https://docs.rs/image/latest/image/ to load and save images

use ndarray::{Array2, ArrayView2};
use image::{ImageBuffer, Rgb};
use palette::Srgb;
use palette::{named::RED, named::BLUE, named::WHITE, named::NAVY, named::MAROON };
use enterpolation::{linear::ConstEquidistantLinear, Curve};
use std::fs::File;
use std::io::{self, Write};



pub fn get_strata_from_gray_scale_image(arg: &str) -> Array2<f64> {
    // Load the PNG file using the `image` crate
    let img = image::open("strata.png").expect("Failed to load image");

    // Convert the image to grayscale and get its pixel data as a 2D array
    let grayscale = img.to_luma8();
    let (width, height) = grayscale.dimensions();
    let data: Vec<f64> = grayscale
        .pixels()
        .map(|p| f64::from(p[0]))
        .collect();
    let array_view = ArrayView2::from_shape((height as usize, width as usize), &data).unwrap();

    // Transpose the array to match the expected orientation of the strata data
    let array = array_view.t().to_owned();

    array
}



pub fn write_seismic_png_image(image: &Array2<f64>, file: &mut File) -> io::Result<()> {
    // Create a gradient from red to blue
    let gradient = ConstEquidistantLinear::<f32, _, 5>::equidistant_unchecked([
        MAROON.into_format(),
        RED.into_format(), 
        WHITE.into_format(),
        BLUE.into_format(),
        NAVY.into_format(),
    ]);
    let gradient_vals = gradient.take(256).collect::<Vec<_>>();

    // Find the average value of the image
    let sum: f64 = image.iter().sum();
    let avg_value = sum / (image.shape()[0] * image.shape()[1]) as f64;

    // Find the min and max values of the image
    let min_value = image.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_value = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // find max difference from average
    let max_diff = f64::max(avg_value - min_value, max_value - avg_value);
    let min_diff = f64::min(avg_value - min_value, max_value - avg_value);



    // Create an empty image buffer
    let (width, height) = (image.shape()[0], image.shape()[1]);
    let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

    // Fill the image buffer with colors based on the input image
    for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
        let value = image[(x as usize, y as usize)];
        let d = max_diff;
        let t = ((d + value - avg_value ) / (2.0*d) * 256.0) as u8;
        // clamp t between 0 and 255
        let t = if t < 0 { 0 } else if t > 255 { 255 } else { t };
        let color = gradient_vals[t as usize];
        let srgb_color: Srgb<u8> = color.into_format();
        *pixel = Rgb([srgb_color.red, srgb_color.green, srgb_color.blue]);
    }

    // Write the image to the file
    img_buffer.write_to(file, image::ImageFormat::Png);

    // Flush the file to ensure all data is written
    file.flush()?;

    Ok(())
}