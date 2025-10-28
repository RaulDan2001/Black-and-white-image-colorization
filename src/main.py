import os
import time
import numpy as np
import cv2
import math
from html import escape
from pathlib import Path

# helpers
def ensure_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def colorize_with_model(net, src_bgr, l_offset=50.0, model_input_size=(224,224)):
    img = src_bgr.astype("float32") / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (model_input_size[0], model_input_size[1]))
    L = cv2.split(resized)[0]
    L_in = L - l_offset
    net.setInput(cv2.dnn.blobFromImage(L_in))
    ab = net.forward()[0].transpose((1,2,0))
    ab = cv2.resize(ab, (src_bgr.shape[1], src_bgr.shape[0]))
    L_orig = cv2.split(lab)[0]
    colorized_lab = np.concatenate((L_orig[:,:,np.newaxis], ab), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr * 255.0, 0, 255).astype("uint8")
    return colorized_bgr

def make_pair_image(gray_bgr, color_bgr):
    h = max(gray_bgr.shape[0], color_bgr.shape[0])
    w = gray_bgr.shape[1] + color_bgr.shape[1]
    pair = np.zeros((h, w, 3), dtype=np.uint8)
    pair[:gray_bgr.shape[0], :gray_bgr.shape[1]] = gray_bgr
    pair[:color_bgr.shape[0], gray_bgr.shape[1]:] = color_bgr
    return pair

# No-reference / proxy metrics
def colorfulness_metric(img_bgr):
    # Hasler and Suesstrunk 2003
    B = img_bgr[:, :, 0].astype('float32')
    G = img_bgr[:, :, 1].astype('float32')
    R = img_bgr[:, :, 2].astype('float32')
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    return math.sqrt(std_rg**2 + std_yb**2) + 0.3 * math.sqrt(mean_rg**2 + mean_yb**2)

def mean_saturation(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype('float32')
    return float(np.mean(s) / 255.0)

def mean_chroma(img_bgr):
    lab = cv2.cvtColor(img_bgr.astype('float32') / 255.0, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt(a**2 + b**2)
    return float(np.mean(chroma))

def chroma_entropy(img_bgr, bins=64):
    lab = cv2.cvtColor(img_bgr.astype('float32') / 255.0, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].ravel()
    b = lab[:, :, 2].ravel()
    chroma = np.sqrt(a**2 + b**2)
    if chroma.size == 0:
        return 0.0
    hist, _ = np.histogram(chroma, bins=bins, range=(chroma.min(), chroma.max()), density=True)
    hist = hist + 1e-12
    ent = -np.sum(hist * np.log(hist))
    return float(ent)

# --- main ---
def main():
    # Resolve script directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Read from env variables
    image_folder = Path(str(repo_root / "dataset"))
    outdir = Path(str(repo_root / "experiments_bw"))
    prototxt = Path(str(repo_root / "models" / "colorization_deploy_v2.prototxt"))
    caffemodel = Path(str(repo_root / "models" / "colorization_release_v2.caffemodel"))
    pts = Path(str(repo_root / "models" / "pts_in_hull.npy"))
    l_offset = 50.0
    size_w = 224
    size_h = 224
    conv8_scale = 2.606

    # Validate existence
    missing = []
    if not prototxt.exists(): missing.append(str(prototxt))
    if not caffemodel.exists(): missing.append(str(caffemodel))
    if not pts.exists(): missing.append(str(pts))
    if not image_folder.exists(): missing.append(str(image_folder))
    if missing:
        print("ERROR: Required files or folders not found:")
        for p in missing:
            print("  -", p)
        print("\nSet environment variables like IMAGE_FOLDER, PROTOTXT, etc., or place files in default locations.")
        return

    os.makedirs(outdir, exist_ok=True)
    imgs_outdir = outdir / "images"
    imgs_outdir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    except cv2.error:
        print("OpenCV failed to load the Caffe model. Paths used:")
        print(" prototxt:", str(prototxt))
        print(" caffemodel:", str(caffemodel))
        raise

    points = np.load(str(pts))
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], conv8_scale, dtype="float32")]

    run_id = time.strftime("%Y%m%d-%H%M%S")

    image_files = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif'))]

    # Prepare simple HTML gallery
    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Colorization results</title></head><body>",
        f"<h1>Colorization results: {escape(run_id)}</h1>",
        "<table border='1' cellpadding='6'><tr><th>Image</th><th>Pair</th><th>colorfulness</th><th>mean_saturation</th><th>mean_chroma</th><th>chroma_entropy</th><th>time_s</th></tr>"
    ]

    for fname in image_files:
        path = Path(image_folder) / fname
        src = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if src is None:
            print("skip (cannot read):", path)
            continue
        src_bgr = ensure_bgr(src)
        gray_for_display = src_bgr.copy()

        start = time.time()
        colorized = colorize_with_model(net, src_bgr, l_offset=l_offset, model_input_size=(size_w, size_h))
        elapsed = time.time() - start

        cf = colorfulness_metric(colorized)
        ms = mean_saturation(colorized)
        mc = mean_chroma(colorized)
        ce = chroma_entropy(colorized)

        pair = make_pair_image(gray_for_display, colorized)
        out_fname = f"{Path(fname).stem}_pair.png"
        out_path = imgs_outdir / out_fname
        cv2.imwrite(str(out_path), pair)

        html_lines.append(
            "<tr>"
            f"<td>{escape(fname)}</td>"
            f"<td><a href='images/{escape(out_fname)}'><img src='images/{escape(out_fname)}' width='420'></a></td>"
            f"<td>{cf:.3f}</td>"
            f"<td>{ms:.3f}</td>"
            f"<td>{mc:.3f}</td>"
            f"<td>{ce:.3f}</td>"
            f"<td>{elapsed:.3f}</td>"
            "</tr>"
        )
        print(f"[{fname}] colorfulness={cf:.3f} sat={ms:.3f} chroma={mc:.3f} time={elapsed:.3f}s")

    html_lines.append("</table></body></html>")
    gallery_path = outdir / f"{run_id}_gallery.html"
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print("done. outputs in", outdir)
    print("Open the gallery:", gallery_path)

if __name__ == "__main__":
    main()