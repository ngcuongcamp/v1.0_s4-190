# Scanner S4-190

- `note`: S4-200 sends signal scan(QR Code)
- `objects`: S4-200(PLC), S4-190, SFC

## Signals:

### `1. Signal S4-200(PLC) send/ S4-190 receive`

- `b'0'`: Failed to scan QRCode(S4-200)
- `b'1'`: Signal scan 3Code
- `b'xxxxxxxxx'`: QRCode
- `b'5'`: Reset program
- `b'9'`: Close program

### `2. Signal S4-190 send/ SFC receive`

- `b'xxxxxxxxx'`: QRCode
- `b'yyyyyyyyy'`: 3Code

### `3. Signal SFC send/ S4-190 receive`

- `b'00'`: Check SFC fail
- `b'01'`: Check SFC pass

### `4. Signal S4-190 send/ S4-200(PLC) receive`

- `b'00'`: Check SFC fail
- `b'01'`: Check SFC pass
- `b'0`: Scan 3Code fail
