extern crate bindgen;
extern crate pkg_config;

use std::env;
use std::path::Path;

fn main() {
    let default_include_dirs = &["/usr/include/dc1394", "/usr/local/include/dc1394"];

    // --------- link to libdc1394 ---------
    pkg_config::find_library("libdc1394-2").unwrap();

    // --------- generate bindings for libdc1394 ---------
    let dir = env::var("DC1394_INCDIR").ok();
    let header = dir.iter()
        .map(|d| &**d)
        .chain(default_include_dirs.iter().map(|v| *v))
        .map(|incdir| Path::new(incdir).join("dc1394.h"))
        .filter(|header| header.is_file())
        .next();

    let header = if let Some(h) = header {
        h
    } else {
        panic!(
            "no header found at {:?}. Set env var DC1394_INCDIR.",
            header
        );
    };

    let mut builder = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .constified_enum_module("dc1394.*_t")
        .whitelist_var("DC1394_(.*)")
        .whitelist_function("dc1394_(.*)");

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
