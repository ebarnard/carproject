#include <SPI.h>

const int NUM_CARS = 2;
const int POT_CS_PINS[NUM_CARS] = {2, 14};
const int REMOTE_POWER_PINS[NUM_CARS] = {3, 19};

void setup() {
  // Set slave-select (SS) pin to output so we don't become an SPI slave device.
  pinMode(10, OUTPUT);
  // Init SPI
  SPI.begin();

  // Initialise the car controllers
  for (int i = 0; i < NUM_CARS; i++) {
    // Turn off the controller
    controller_power(i, 0);
    pinMode(REMOTE_POWER_PINS[i], OUTPUT);
  
    // Disable POT_0 SPI chip select.
    digitalWrite(POT_CS_PINS[i], HIGH);
    pinMode(POT_CS_PINS[i], OUTPUT);

    // Set the POTs to middle range
    send_command(i, 128, 128);
  }

  // Start USB serial interface
  Serial.begin(115200);
  Serial.println("KYOSHO CAR CONTROLLER INTERFACE");
  Serial.println("");
  command_help();
}

void loop() {
  int car = 0;
  int throttle = 0;
  int steering = 0;

  if (!serial_read_u8(&car)) {
    return error("expected car number");
  }
  if (!serial_read_space()) {
    return error("expected space after car number");
  }
  if (serial_is_number()) {
    if (!serial_read_u8(&throttle)) {
      return error("expected throttle value");
    }
    if (!serial_read_space()) {
      return error("expected space after throttle value");
    }
    if (!serial_read_u8(&steering)) {
      return error("expected steering value");
    }
    if (!serial_read_newline()) {
      return error("expected newline after steering value");
    }
    send_command(car, throttle, steering);
  } else {
    // Check for ON/OFF command
    char buf[4];
    int buf_len = serial_read_string(buf, sizeof(buf));

    if (string_eq(buf, buf_len, "ON", 2)) {
      if (serial_read_newline()) {
        return controller_power(car, 1);
      } else {
        return error("expected newline after ON command");
      }
    } else if (string_eq(buf, buf_len, "OFF", 3)) {
      if (serial_read_newline()) {
        return controller_power(car, 0);
      } else {
        return error("expected newline after OFF command");
      }
    }
    return error("expected ON or OFF command");
  }
}

int serial_peek_byte() {
  while (Serial.available() == 0) { }
  return Serial.peek();
}

int serial_read_byte() {
  while (Serial.available() == 0) { }
  return Serial.read();
}

int serial_read_string(char* buf, int buf_len) {
  int pos = 0;
  
  while (pos < buf_len) {
    int serial_byte = serial_peek_byte();
    if (serial_byte == ' ' || serial_byte == '\r' || serial_byte == '\n') {
      break;
    }
    buf[pos] = serial_read_byte();
    pos += 1;
  }

  return pos;
}

bool string_eq(const char* str1, int str1_len, const char* str2, int str2_len) {
  if (str1_len != str2_len) {
    return false;
  }
  for (int i = 0; i < str1_len; i++) {
    if (str1[i] != str2[i]) {
      return false;
    }
  }
  return true;
}

bool serial_is_number() {
  int serial_byte = serial_peek_byte();
  if (serial_byte >= '0' && serial_byte <= '9') {
    return true;
  } else {
    return false;
  }
}

bool serial_read_u8(int* out) {
  char buf[5];
  int buf_len = serial_read_string(buf, sizeof(buf));

  if (buf_len == sizeof(buf)) {
    Serial.print("error: 8-bit integer string too long");
    return false;
  }

  int pos = 0;
  bool neg = false;
  
  if (buf[pos] == '-') {
    neg = true;
    pos += 1;
  }

  *out = 0;
  while (pos < buf_len) {
    int serial_byte = buf[pos];
    if (serial_byte >= '0' && serial_byte <= '9') {
      int value = serial_byte - '0';
      *out = *out * 10 + value;
      
      // dont't let out overflow
      if (*out > 255) {
        Serial.print("error: 8-bit integer overflow");
        return false;
      }
    } else {
      Serial.print("error: not integer character encountered");
      return false;
    }
    
    pos += 1;
  }

  if (neg) {
    *out *= -1;
  }
  return true;
}

bool serial_read_space() {
  if (serial_peek_byte() == ' ') {
    serial_read_byte();
    return true;
  }
  return false;
}

bool serial_read_newline() {
  int serial_byte = serial_peek_byte();
  if (serial_byte == '\n') {
    serial_read_byte();
    return true;
  } else if (serial_byte == '\r') {
    serial_read_byte();
    if (serial_peek_byte() == '\n') {
      serial_read_byte();
      return true;
    }
  }
  return false;
}

void command_help() {
  Serial.println("commands should be of the form \"{car number} {speed} {steering}\\n\"");
}

void error(const char* message) {  
  Serial.print("error: ");
  Serial.println(message);
  command_help();

  // Wait for the rest of any commands to be sent
  delay(250);
  
  // Clear serial buffer
  while (Serial.read() != -1) { }
}

void controller_power(int car, int on) {
  if (car < 0 || car >= NUM_CARS) {
    Serial.print("error: car number ");
    Serial.print(car);
    Serial.println(" does not exist");
    return;
  }
  
  if (on) {
    digitalWrite(REMOTE_POWER_PINS[car], HIGH);
  } else {
    digitalWrite(REMOTE_POWER_PINS[car], LOW);
  }
}

void send_command(int car, int throttle, int steering) {
  if (car < 0 || car >= NUM_CARS) {
    Serial.print("error: car number ");
    Serial.print(car);
    Serial.println(" does not exist");
    return;
  }

  SPI.beginTransaction(SPISettings(14000000, MSBFIRST, SPI_MODE0));
  // Enable POT_0 SPI interface.
  digitalWrite(POT_CS_PINS[car], LOW);
  // Wait at least 60ns for it to become enabled
  __asm__("nop\n\t");
  __asm__("nop\n\t");
  __asm__("nop\n\t");

  SPI.transfer(0x00);
  SPI.transfer(throttle);
  SPI.transfer(0x10);
  SPI.transfer(steering);

  // Disable it again.
  digitalWrite(POT_CS_PINS[car], HIGH);
  SPI.endTransaction();
}

