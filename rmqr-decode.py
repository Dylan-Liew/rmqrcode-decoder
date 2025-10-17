import os
import sys
from typing import List, Tuple
from PIL import Image
from rmqrcode.format.rmqr_versions import rMQRVersions
from rmqrcode.format.alignment_pattern_coordinates import AlignmentPatternCoordinates
from rmqrcode.format.mask import mask as mask_fn
from rmqrcode.format.error_correction_level import ErrorCorrectionLevel
from rmqrcode.encoder import ByteEncoder

QUIET_ZONE_MODULES = 2
THRESH = 128

def _binarize(img: Image.Image) -> Image.Image:
    return img.convert("L")


def _first_black_pixel(bw: Image.Image) -> Tuple[int, int]:
    px = bw.load()
    for y in range(bw.size[1]):
        for x in range(bw.size[0]):
            if px[x, y] < THRESH:
                return x, y
    raise ValueError("No dark modules found")


def _estimate_module_and_quiet(bw: Image.Image) -> Tuple[int, int, int]:
    px = bw.load()
    w, h = bw.size
    x1, y1 = _first_black_pixel(bw)
    x = x1
    while x < w and px[x, y1] < THRESH:
        x += 1
    run_h = x - x1
    y = y1
    while y < h and px[x1, y] < THRESH:
        y += 1
    run_v = y - y1
    module_size = max(1, int(round(min(run_h, run_v) / 7.0)))
    qmods_x = int(round(x1 / module_size))
    qmods_y = int(round(y1 / module_size))
    qmods = int(round((qmods_x + qmods_y) / 2)) or QUIET_ZONE_MODULES
    return module_size, qmods, qmods


def _symbol_size_from_image(bw: Image.Image, module_size: int, qx: int, qy: int) -> Tuple[int, int]:
    gw = int(round(bw.size[0] / module_size))
    gh = int(round(bw.size[1] / module_size))
    return gw - 2 * qx, gh - 2 * qy


def _sample(bw: Image.Image, module_size: int, qx: int, qy: int, x: int, y: int) -> int:
    cx = min(max(int((qx + x + 0.5) * module_size), 0), bw.size[0] - 1)
    cy = min(max(int((qy + y + 0.5) * module_size), 0), bw.size[1] - 1)
    return 1 if bw.getpixel((cx, cy)) < THRESH else 0


def _reserved_mask(width: int, height: int) -> List[List[bool]]:
    R = [[False for _ in range(width)] for __ in range(height)]
    for i in range(7):
        for j in range(7):
            R[i][j] = True
    for n in range(8):
        if n < height:
            R[n][7] = True
        if height >= 9:
            R[7][n] = True
    for i in range(5):
        for j in range(5):
            R[height - 1 - i][width - 1 - j] = True
    R[height - 1][0] = R[height - 1][1] = R[height - 1][2] = True
    if height >= 11:
        R[height - 2][0] = R[height - 2][1] = True
    R[0][width - 1] = R[0][width - 2] = R[1][width - 1] = True
    R[1][width - 2] = True
    centers = AlignmentPatternCoordinates[width]
    for cx in centers:
        for i in range(3):
            for j in range(3):
                R[i][cx + j - 1] = True
                R[height - 1 - i][cx + j - 1] = True
    for j in range(width):
        for i in (0, height - 1):
            R[i][j] = True if not R[i][j] else True
    for i in range(height):
        for j in [0, width - 1] + centers:
            R[i][j] = True if not R[i][j] else True
    si, sj = 1, 8
    for n in range(18):
        di = n % 5
        dj = n // 5
        R[si + di][sj + dj] = True
    si, sj = height - 1 - 5, width - 1 - 7
    for n in range(15):
        di = n % 5
        dj = n // 5
        R[si + di][sj + dj] = True
    R[height - 1 - 5][width - 1 - 4] = True
    R[height - 1 - 5][width - 1 - 3] = True
    R[height - 1 - 5][width - 1 - 2] = True
    return R


def _collect_bits(bw: Image.Image, w: int, h: int, ms: int, qx: int, qy: int, codewords_total: int, remainder_bits: int) -> List[int]:
    reserved = _reserved_mask(w, h)
    expected = codewords_total * 8 + remainder_bits
    dy = -1
    cx, cy = w - 2, h - 6
    bits: List[int] = []
    while cx >= 0 and len(bits) < expected:
        for x in (cx, cx - 1):
            if 0 <= x < w and 0 <= cy < h and not reserved[cy][x]:
                val = _sample(bw, ms, qx, qy, x, cy)
                if mask_fn(x, cy):
                    val ^= 1
                bits.append(val)
                if len(bits) == expected:
                    break
        if dy < 0 and cy == 1:
            cx -= 2
            dy = 1
        elif dy > 0 and cy == h - 2:
            cx -= 2
            dy = -1
        else:
            cy += dy
    return bits[:-remainder_bits] if remainder_bits else bits


def _deinterleave_data(final_codewords: List[str], blocks_def: List[dict]) -> List[str]:
    blocks = []
    for bd in blocks_def:
        for _ in range(bd["num"]):
            blocks.append({"k": bd["k"], "ecc": bd["c"] - bd["k"], "data": [], "par": []})
    ptr = 0
    max_k = max(b["k"] for b in blocks)
    for i in range(max_k):
        for b in blocks:
            if i < b["k"]:
                b["data"].append(final_codewords[ptr]); ptr += 1
    max_e = max(b["ecc"] for b in blocks)
    for i in range(max_e):
        for b in blocks:
            if i < b["ecc"]:
                # parity not used in this simple decoder
                ptr += 1
    data_codewords: List[str] = []
    for b in blocks:
        data_codewords.extend(b["data"])
    return data_codewords


def _decode_byte_segment(data_codewords: List[str], version_name: str) -> str:
    bits = "".join(data_codewords)
    if len(bits) < 3 or bits[:3] != ByteEncoder.mode_indicator():
        raise NotImplementedError("Only Byte mode supported.")
    cci_len = rMQRVersions[version_name]["character_count_indicator_length"][ByteEncoder]
    if len(bits) < 3 + cci_len:
        raise ValueError("Bitstream too short")
    count = int(bits[3:3 + cci_len], 2)
    payload = bits[3 + cci_len:3 + cci_len + count * 8]
    if len(payload) != count * 8:
        raise ValueError("Bitstream shorter than declared byte count")
    return bytes(int(payload[i:i + 8], 2) for i in range(0, len(payload), 8)).decode("utf-8")


def decode_png(path: str) -> str:
    bw = _binarize(Image.open(path))
    module_size, qx, qy = _estimate_module_and_quiet(bw)
    sym_w, sym_h = _symbol_size_from_image(bw, module_size, qx, qy)
    version_name = f"R{sym_h}x{sym_w}"
    if version_name not in rMQRVersions:
        raise ValueError(f"Unsupported rMQR size: {version_name}")
    ver = rMQRVersions[version_name]
    bits = _collect_bits(bw, sym_w, sym_h, module_size, qx, qy, ver["codewords_total"], ver["remainder_bits"])
    s = "".join("1" if b else "0" for b in bits)
    final_codewords = [s[i:i + 8] for i in range(0, len(s), 8)]
    data_codewords = _deinterleave_data(final_codewords, ver["blocks"][ErrorCorrectionLevel.M])
    return _decode_byte_segment(data_codewords, version_name)


def main():
    if len(sys.argv) != 2:
        print("Usage: python rmqr-decode.py <image.png>", file=sys.stderr)
        sys.exit(1)
    try:
        print(decode_png(sys.argv[1]))
    except Exception as e:
        print(f"Decode error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
