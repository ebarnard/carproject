extern crate libdc1394_sys;
extern crate triple_buffer;

use self::libdc1394_sys::*;

use std::mem;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

pub struct Camera {
    dc: *mut dc1394_t,
    dc_cam: *mut dc1394camera_t,
}

impl Camera {
    pub fn new() -> Camera {
        unsafe {
            let d = dc1394_new();
            if d == 0 as *mut _ {
                panic!("could not create dc1394 instance");
            }

            let mut list = 0 as *mut _;
            let err = dc1394_camera_enumerate(d, &mut list);
            if err != dc1394error_t::DC1394_SUCCESS {
                panic!("could not enumerate cameras")
            }

            if (*list).num == 0 {
                panic!("no cameras found");
            }

            let camera = dc1394_camera_new(d, (*(*list).ids).guid);
            if camera == 0 as *mut _ {
                panic!("failed to initialize camera");
            }

            dc1394_camera_free_list(list);

            //printf("Using camera with GUID %"PRIx64"\n", camera->guid);

            Camera {
                dc: d,
                dc_cam: camera,
            }
        }
    }

    pub fn start_capture(&mut self) -> Capture {
        unsafe {
            /*-----------------------------------------------------------------------
             *  get the best video mode and highest framerate. This can be skipped
             *  if you already know which mode/framerate you want...
             *-----------------------------------------------------------------------*/
            // get video modes:
            let mut video_modes = mem::zeroed();
            let err = dc1394_video_get_supported_modes(self.dc_cam, &mut video_modes);
            if err != dc1394error_t::DC1394_SUCCESS {
                panic!("failed to get supported video modes");
            }

            // select highest res mode:
            let mut best_video_mode = 0;

            for i in (0..video_modes.num).rev() {
                let video_mode = video_modes.modes[i as usize];
                if dc1394_is_video_mode_scalable(video_mode) == 0 {
                    let mut coding = 0;
                    dc1394_get_color_coding_from_video_mode(self.dc_cam, video_mode, &mut coding);
                    if coding == dc1394color_coding_t::DC1394_COLOR_CODING_MONO8 {
                        best_video_mode = video_mode;
                        break;
                    }
                }
            }

            if best_video_mode == 0 {
                panic!("could not get a valid MONO8 mode");
            }

            // get highest framerate
            let mut framerates = mem::zeroed();

            let err = dc1394_video_get_supported_framerates(
                self.dc_cam,
                best_video_mode,
                &mut framerates,
            );
            if err != dc1394error_t::DC1394_SUCCESS {
                panic!("could not get framerates");
            }

            let best_framerate = framerates.framerates[framerates.num as usize - 1];

            /*-----------------------------------------------------------------------
             *  setup capture
             *-----------------------------------------------------------------------*/

            let err = dc1394_video_set_iso_speed(self.dc_cam, dc1394speed_t::DC1394_ISO_SPEED_400);
            if err != dc1394error_t::DC1394_SUCCESS {
                panic!("could not set iso speed");
            }

            let err = dc1394_video_set_mode(self.dc_cam, best_video_mode);
            if err != dc1394error_t::DC1394_SUCCESS {
                panic!("could not set video mode");
            }

            let err = dc1394_video_set_framerate(self.dc_cam, best_framerate);
            if err != dc1394error_t::DC1394_SUCCESS {
                panic!("could not set framerate");
            }

            let err = dc1394_capture_setup(self.dc_cam, 4, DC1394_CAPTURE_FLAGS_DEFAULT);
            if err != dc1394error_t::DC1394_SUCCESS {
                panic!("could not setup camera. make sure that the video mode and framerate are supported by your camera");
            }

            /*-----------------------------------------------------------------------
             *  report camera's features
             *-----------------------------------------------------------------------*/
            /*let mut features = mem::zeroed();
            let err = dc1394_feature_get_all(camera, &mut features);
            if err != dc1394error_t::DC1394_SUCCESS {
                panic!("Could not get feature set");
            }*/

            //dc1394_feature_print_all(&mut features, stdout);

            let mut width = 0;
            let mut height = 0;
            dc1394_get_image_size_from_video_mode(
                self.dc_cam,
                best_video_mode,
                &mut width,
                &mut height,
            );

            // have the camera start sending us data
            let err = dc1394_video_set_transmission(self.dc_cam, dc1394switch_t::DC1394_ON);
            if err != dc1394error_t::DC1394_SUCCESS {
                panic!("could not start camera transmission");
            }

            let mut inital_frame = vec![0; width as usize * height as usize];
            capture_frame(self.dc_cam, width, height, &mut inital_frame);
            let (mut input, output) = triple_buffer::TripleBuffer::new(inital_frame).split();

            let capture_thread_stop = Arc::new(AtomicBool::new(false));
            let capture_thread_stop_clone = capture_thread_stop.clone();

            let dc_cam_usize = self.dc_cam as usize;
            let capture_thread_handle = thread::spawn(move || {
                let dc_cam = dc_cam_usize as *mut _;

                while !capture_thread_stop_clone.load(Ordering::Relaxed) {
                    capture_frame(dc_cam, width, height, input.raw_input_buffer());
                    input.raw_publish();
                }

                // stop data transmission
                let err = dc1394_video_set_transmission(dc_cam, dc1394switch_t::DC1394_OFF);
                if err != dc1394error_t::DC1394_SUCCESS {
                    println!("capture drop error {}", err);
                    // TODO: is panic worth it?
                    panic!("could not stop the camera");
                }
            });

            Capture {
                _camera: self,
                capture_thread_stop,
                capture_thread_handle: Some(capture_thread_handle),
                output,
                width,
                height,
            }
        }
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        unsafe {
            dc1394_capture_stop(self.dc_cam);
            dc1394_camera_free(self.dc_cam);
            dc1394_free(self.dc);
        }
    }
}

pub struct Capture<'a> {
    _camera: &'a mut Camera,
    capture_thread_stop: Arc<AtomicBool>,
    capture_thread_handle: Option<JoinHandle<()>>,
    output: triple_buffer::Output<Vec<u8>>,
    width: u32,
    height: u32,
}

impl<'a> Capture<'a> {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn bytes(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn latest_frame(&mut self) -> &[u8] {
        &self.output.read()
    }

    pub fn stop(self) {
        drop(self);
    }
}

impl<'a> Drop for Capture<'a> {
    fn drop(&mut self) {
        self.capture_thread_stop.store(true, Ordering::SeqCst);
        self.capture_thread_handle
            .take()
            .unwrap()
            .join()
            .expect("capture thread panicked");
    }
}

unsafe fn capture_frame(dc_cam: *mut dc1394camera_t, width: u32, height: u32, buf: &mut [u8]) {
    // capture one frame
    let mut frame = 0 as *mut _;
    let err = dc1394_capture_dequeue(
        dc_cam,
        dc1394capture_policy_t::DC1394_CAPTURE_POLICY_WAIT,
        &mut frame,
    );
    if err != dc1394error_t::DC1394_SUCCESS {
        println!("deque error {}", err);
        panic!("could not deque frame");
    }

    if (*frame).size != [width, height] {
        panic!("frame not expected size");
    }

    if (*frame).stride != width {
        panic!("strides not supported")
    }

    // copy data to buf
    let frame_image_len = (width * height) as usize;
    if buf.len() != frame_image_len {
        panic!("buf wrong size");
    }
    assert_eq!(frame_image_len as u64, (*frame).image_bytes as u64);
    assert_eq!((*frame).padding_bytes, 0);

    let data = slice::from_raw_parts((*frame).image, frame_image_len);
    buf.copy_from_slice(data);

    // return frame to ringbuffer
    let err = dc1394_capture_enqueue(dc_cam, frame);
    if err != dc1394error_t::DC1394_SUCCESS {
        panic!("could not return frame");
    }
}
