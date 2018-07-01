extern crate cmake;

fn main() {
    let dst = cmake::Config::new("cpp").build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=dylib=opencv_wrapper");
}
