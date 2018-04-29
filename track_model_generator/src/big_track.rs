use {CircleS, Direction};

pub const WIDTH: f64 = 0.465;
pub const SCALE: f64 = 0.3;

pub const DEF: &'static [CircleS] = &[CircleS {
                                          x: 8.0 * SCALE,
                                          y: 6.0 * SCALE,
                                          r: SCALE,
                                          d: Direction::Anticlockwise,
                                      },
                                      CircleS {
                                          x: 2.0 * SCALE,
                                          y: 6.0 * SCALE,
                                          r: SCALE,
                                          d: Direction::Anticlockwise,
                                      },
                                      CircleS {
                                          x: 2.0 * SCALE,
                                          y: 2.0 * SCALE,
                                          r: SCALE,
                                          d: Direction::Anticlockwise,
                                      },
                                      CircleS {
                                          x: 4.0 * SCALE,
                                          y: 4.0 * SCALE,
                                          r: SCALE,
                                          d: Direction::Clockwise,
                                      },
                                      CircleS {
                                          x: 6.0 * SCALE,
                                          y: 2.0 * SCALE,
                                          r: SCALE,
                                          d: Direction::Anticlockwise,
                                      },
                                      CircleS {
                                          x: 8.0 * SCALE,
                                          y: 2.0 * SCALE,
                                          r: SCALE,
                                          d: Direction::Anticlockwise,
                                      },
                                      CircleS {
                                          x: 8.0 * SCALE,
                                          y: 4.0 * SCALE,
                                          r: SCALE,
                                          d: Direction::Clockwise,
                                      }];
