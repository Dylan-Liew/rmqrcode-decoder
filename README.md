# Simple rMQR decoder

Requirements:
```
pip install pillow rmqrcode
```

Usage:
```
python rmqr-decode.py image.png
```

Limitations:
- Image has a 2‑module quiet zone, no rotation/skew, and no scaling (crisp PNG).
- ECC = M and a single Byte‑mode segment.
- No error correction during decode; expects clean images from a compatible rMQR encoder.


