extern crate cmake;

fn main() {
    let dst = cmake::Config::new("cxx").uses_cxx11().build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=dylib=track_aruco");
}
