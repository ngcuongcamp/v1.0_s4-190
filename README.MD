## Scanner S4-190

- `objects`: S4-200(PLC), S4-190, SFC

### Signals:

#### _`1. Signal S4-200(PLC) sends/ S4-190 receives`_

- `b'0'`: Failed to scan QRCode(S4-200)
- `b'1'`: Signal scan 3Code
- `b'xxxxxxxxx'`: QRCode
- `b'5'`: Reset program
- `b'9'`: Close program

#### _`2. Signal S4-190 sends/ SFC receives`_

- `b'xxxxxxxxx'`: QRCode
- `b'yyyyyyyyy'`: 3Code

#### _`3. Signal SFC sends/ S4-190 receives`_

- `b'0'`: Check SFC fail
- `b'1'`: Check SFC pass

#### _`4. Signal S4-190 sends/ S4-200(PLC) receives`_

- `b'0'`: Check SFC fail
- `b'1'`: Check SFC pass
- `b'2`: Scan 3Code fail
