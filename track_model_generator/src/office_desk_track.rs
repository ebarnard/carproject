use {CircleS, Direction};

pub const WIDTH: f64 = 0.465;

pub const DEF: &'static [CircleS] = &[CircleS {
                                          x: 0.3,
                                          y: -0.3,
                                          r: 0.3,
                                          d: Direction::Anticlockwise,
                                      },
                                      CircleS {
                                          x: 0.3,
                                          y: 0.3,
                                          r: 0.3,
                                          d: Direction::Anticlockwise,
                                      },
                                      CircleS {
                                          x: -0.3,
                                          y: 0.3,
                                          r: 0.3,
                                          d: Direction::Anticlockwise,
                                      },
                                      CircleS {
                                          x: -0.3,
                                          y: -0.3,
                                          r: 0.3,
                                          d: Direction::Clockwise,
                                      }];
