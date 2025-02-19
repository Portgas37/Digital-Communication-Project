
# Digital Communication Project

## Overview
This project implements a digital communication system to reliably transmit text messages over a noisy channel. It is designed as part of the EPFL course **Principles of Digital Communications (COM-302)**.

---

## Channel Model
The channel has two states, B = 0 and B = 1, each with equal probability. The input vector x is mapped to an output vector Y as follows:

    Y = 
        [0, I_n] x + Z     if B = 0
        [I_n, 0] x + Z     if B = 1

- Z ~ N(0, 25I_{2n}) is Gaussian noise.
- Input constraints:
  - Maximum block length: n ≤ 500,000
  - Maximum energy: || x ||^2 ≤ 40,960

---

## Objective
The goal is to design a communication system that:
- **Transmitter**: Encodes a 40-character text message into a real-valued signal.
- **Channel**: Transmits the encoded signal through the noisy channel.
- **Receiver**: Decodes the received signal to reconstruct the original message with minimal errors.

---

## Transmitter Design
This project implements several encoding schemes for efficient communication:
- `transmitter0`: Bit-by-bit encoding (low energy, high error rate)
- `transmitter1`: Letter-by-letter encoding (no error, high energy)
- `transmitter2`, `transmitter3`, `transmitter4`: Multi-letter encoding for improved efficiency and reduced error rates

---

## Receiver Design
The receiver:
- Applies **Maximum Likelihood Decoding** to reconstruct the message.
- Uses statistical analysis (e.g., **Normality Tests**) to identify the channel state and accurately decode the transmitted signal.

---

## Usage
To run the client locally, use:

    python3 client.py --input_file=input.txt --output_file=output.txt --srv_hostname=iscsrv72.epfl.ch --srv_port=80

### Example:
    python3 client.py --input_file example_input.txt --output_file example_output.txt --srv_hostname=iscsrv72.epfl.ch --srv_port=80

---

## Dependencies
This project requires:
- Python 3.x
- NumPy
- SciPy

Install dependencies with:
    pip install numpy scipy

---

## Testing Locally
A local version of the channel is provided for testing:

    from channel_helper import channel

This allows you to test the transmitter and receiver without connecting to the EPFL server.

---

## Performance Metrics
- **Message Length**: 40 characters
- **Maximum Input Energy**: 40,960
- **Error Metrics**: Character error rate and message error rate

