import cv2
import numpy as np
import os
import gc
import sys
import json
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pkg_resources
import argparse


from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


from segment_anything import SamPredictor, sam_model_registry, sam_hq_model_registry

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
INTERMEDIATE_PREFIXES = ("raw_", "grounded_sam_output_", "mask_")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    path = str(path)
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image):
    path = str(path)
    ext = Path(path).suffix
    if not ext:
        raise ValueError(f"Output path missing extension: {path}")
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False
    encoded.tofile(path)
    return True


def collect_image_paths(root_dir, suffixes=IMAGE_SUFFIXES, recursive=True):
    root = Path(root_dir)
    if not root.exists():
        return []
    if recursive:
        return [str(p) for p in root.rglob('*') if p.is_file() and p.suffix.lower() in suffixes]
    return [str(p) for p in root.iterdir() if p.is_file() and p.suffix.lower() in suffixes]


def iter_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = [get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer) for logit in logits_filt]
    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
    mask_image = mask.reshape(mask.shape[0], mask.shape[1], 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)

def save_mask_data(mask_image_dir, json_dir, mask_list, box_list, label_list, base_name):
\
\
\

    ensure_dir(mask_image_dir)
    ensure_dir(json_dir)
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    mask_img_path = os.path.join(mask_image_dir, f"mask_{base_name}.jpg")
    plt.savefig(mask_img_path, bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{'value': value, 'label': 'background'}]
    for label, box in zip(label_list, box_list):
        if '(' in label and ')' in label:
            name, logit = label.split('(')
            logit = logit[:-1]
        else:
            name = label
            logit = "1.0"
        json_data.append({'value': value + 1, 'label': name, 'logit': float(logit), 'box': box.numpy().tolist()})
        value += 1
    mask_json_path = os.path.join(json_dir, f"mask_{base_name}.json")
    with open(mask_json_path, 'w') as f:
        json.dump(json_data, f)

def create_cutout_image(original_image, masks, image_file, output_dir):
    image_array = np.array(original_image)
    combined_mask = torch.zeros(masks.shape[-2:], dtype=torch.bool)
    for mask in masks:
        combined_mask = combined_mask | mask.cpu().squeeze(0)
    combined_mask_np = combined_mask.numpy()

    cutout_image = np.zeros_like(image_array)
    cutout_image[combined_mask_np] = image_array[combined_mask_np]
    cutout_pil = Image.fromarray(cutout_image)
    cutout_pil.save(os.path.join(output_dir, image_file))
    return cutout_image

def create_cutout_with_transparency(original_image, masks, image_file, output_dir):
    image_array = np.array(original_image)
    combined_mask = torch.zeros(masks.shape[-2:], dtype=torch.bool)
    for mask in masks:
        combined_mask = combined_mask | mask.cpu().squeeze(0)
    combined_mask_np = combined_mask.numpy()

    h, w = image_array.shape[:2]
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_image[:, :, :3] = image_array
    rgba_image[:, :, 3] = combined_mask_np.astype(np.uint8) * 255

    ensure_dir(output_dir)
    rgba_pil = Image.fromarray(rgba_image, 'RGBA')
    stem = Path(image_file).stem
    rgba_pil.save(os.path.join(output_dir, f"{stem}_transparent.png"))
    return rgba_image

def process_white_background(image_path, output_folder, out_filename=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if out_filename is None:
        image_file = os.path.basename(image_path)
    else:
        image_file = out_filename
    image = imread_unicode(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white_gray1 = np.array([0, 0, 180])
    upper_white_gray1 = np.array([180, 30, 255])
    lower_white_gray2 = np.array([0, 0, 120])
    upper_white_gray2 = np.array([180, 40, 200])
    mask_white_gray1 = cv2.inRange(hsv, lower_white_gray1, upper_white_gray1)
    mask_white_gray2 = cv2.inRange(hsv, lower_white_gray2, upper_white_gray2)
    mask = mask_white_gray1 | mask_white_gray2
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask_final = np.zeros_like(binary)
        cv2.drawContours(mask_final, [largest_contour], -1, 255, thickness=cv2.FILLED)
        final_result = cv2.bitwise_and(result, result, mask=mask_final)
        output_image_path = os.path.join(output_folder, image_file)
        imwrite_unicode(output_image_path, final_result)
        print(f"Processed and saved: {output_image_path}")
    else:
        print(f"No contours found in: {image_file}")

def process_red_background(image_path, output_path, min_ratio=0.3, min_fg_pixels=50, min_area_small=10):
    image = imread_unicode(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask_red1 | mask_red2

    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    fg_pixel_before = np.count_nonzero(binary)
    if fg_pixel_before == 0:
        print("No foreground at all, skip.")
        imwrite_unicode(output_path, np.zeros_like(image))
        return


    kernel = np.ones((4, 4), np.uint8)
    binary_morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_pixel_after = np.count_nonzero(binary_morph)
    ratio = fg_pixel_after / fg_pixel_before if fg_pixel_before else 0


    if ratio < min_ratio or fg_pixel_after < min_fg_pixels:
        print(f"Warning: Morph kills too much foreground (ratio={ratio:.3f}, fg={fg_pixel_after}), skip morph, use raw binary.")
        binary_morph = binary
        fg_pixel_after = fg_pixel_before


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_morph, connectivity=8)
    cleaned = np.zeros_like(binary_morph)


    if fg_pixel_after < 300:
        min_area = min_area_small
    else:
        min_area = 50

    kept = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
            kept += 1
    if kept == 0:
        print("Nothing kept after filtering, outputting all foreground.")
        cleaned = binary_morph


    norm_mask = normalize_mask(cleaned, (256, 256))

    output_path_obj = Path(output_path)
    norm_mask_path = output_path_obj.with_name(f"{output_path_obj.stem}_normmask.png")
    imwrite_unicode(str(norm_mask_path), norm_mask)

    final_result = cv2.bitwise_and(image, image, mask=cleaned)
    imwrite_unicode(output_path, final_result)
    print(f"Processed and saved: {output_path}")


def normalize_mask(mask, out_shape=(256,256)):


    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(out_shape, dtype=np.uint8)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    crop = mask[y_min:y_max+1, x_min:x_max+1]

    h, w = crop.shape
    scale = min(out_shape[0]/h, out_shape[1]/w)
    crop_resized = cv2.resize(crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)

    result = np.zeros(out_shape, dtype=np.uint8)
    new_h, new_w = crop_resized.shape
    start_y = (out_shape[0] - new_h) // 2
    start_x = (out_shape[1] - new_w) // 2
    result[start_y:start_y+new_h, start_x:start_x+new_w] = crop_resized
    return result

def detect_background_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2
    red_ratio = np.sum(mask_red) / (image.shape[0] * image.shape[1])
    if red_ratio > 0.5:
        return "red"
    else:
        return "white"

def postprocess_all(output_root, processed_root, temp_root=None):
    output_root = Path(output_root)
    processed_root = Path(processed_root)

    temp_root_path = Path(temp_root) if temp_root else processed_root
    cutout_files = [p for p in output_root.rglob('*.jpg') if p.is_file()]
    for cutout_path in cutout_files:

        if cutout_path.name.startswith(INTERMEDIATE_PREFIXES):
            continue
        rel_path = cutout_path.relative_to(output_root)
        processed_dir = processed_root / rel_path.parent
        ensure_dir(processed_dir)
        cutout_img = imread_unicode(str(cutout_path))
        if cutout_img is None:
            print(f"Failed to read cutout: {cutout_path}")
            continue
        background_color = detect_background_color(cutout_img)
        if background_color == "white":
            process_white_background(str(cutout_path), str(processed_dir), out_filename=rel_path.name)
        elif background_color == "red":

            temp_dir = temp_root_path / rel_path.parent
            ensure_dir(temp_dir)
            temp_red_out = temp_dir / f"temp_red_{cutout_path.stem}.jpg"
            process_red_background(str(cutout_path), str(temp_red_out))
            process_white_background(str(temp_red_out), str(processed_dir), out_filename=rel_path.name)


def process_batch(batch_files, config):

    input_root = Path(config['image_path']).resolve()
    output_root = Path(config['output_dir'])
    processed_root = Path(config['processed_dir'])

    json_root = Path(config.get('json_root') or config['output_dir'])
    grounded_root = Path(config.get('grounded_root') or config['output_dir'])
    transparent_root = Path(config.get('transparent_root') or config['output_dir'])
    raw_root = Path(config.get('raw_root') or config['output_dir'])
    mask_root = Path(config.get('mask_root') or config['output_dir'])
    model = load_model(
        config['config_file'],
        config['grounded_checkpoint'],
        device=config['device']
    )
    if config['use_sam_hq']:
        predictor = SamPredictor(
            sam_hq_model_registry[config['sam_version']](
                checkpoint=config['sam_hq_checkpoint']
            ).to(config['device'])
        )
    else:
        predictor = SamPredictor(
            sam_model_registry[config['sam_version']](
                checkpoint=config['sam_checkpoint']
            ).to(config['device'])
        )
    for image_path in batch_files:
        try:
            image_path_obj = Path(image_path)
            try:
                rel_path = image_path_obj.resolve().relative_to(input_root)
            except ValueError:
                rel_path = Path(image_path_obj.name)
            image_file = rel_path.name
            image_stem = Path(image_file).stem
            output_dir = output_root / rel_path.parent
            processed_dir = processed_root / rel_path.parent
            json_dir = json_root / rel_path.parent
            mask_dir = mask_root / rel_path.parent
            grounded_dir = grounded_root / rel_path.parent
            transparent_dir = transparent_root / rel_path.parent
            raw_dir = raw_root / rel_path.parent
            ensure_dir(output_dir)
            ensure_dir(processed_dir)
            ensure_dir(json_dir)
            ensure_dir(mask_dir)
            ensure_dir(grounded_dir)
            ensure_dir(transparent_dir)
            ensure_dir(raw_dir)

            imgfile = str(image_path_obj)
            image_pil, image = load_image(imgfile)

            image_pil.save(os.path.join(str(raw_dir), f"raw_{image_stem}.jpg"))
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, config['text_prompt'], config['box_threshold'], config['text_threshold'], device=config['device']
            )
            image_cv = imread_unicode(imgfile)
            if image_cv is None:
                print(f"Failed to read image: {imgfile}")
                continue
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]
            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes_filt, image_rgb.shape[:2]
            ).to(config['device'])
            with torch.no_grad():
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(config['device']),
                    multimask_output=False
                )
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            for mask in masks:
                show_mask(mask.cpu().numpy().squeeze(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                show_box(box.numpy(), plt.gca(), label)
            plt.axis('off')

            plt.savefig(os.path.join(str(grounded_dir), f"grounded_sam_output_{image_stem}.jpg"),
                        bbox_inches="tight", dpi=300, pad_inches=0.0)

            save_mask_data(str(mask_dir), str(json_dir), masks, boxes_filt, pred_phrases, image_stem)
            if config['create_cutout']:
                create_cutout_image(image_rgb, masks, image_file, str(output_dir))
            if config['create_transparent']:

                create_cutout_with_transparency(image_rgb, masks, image_file, str(transparent_dir))

            plt.close('all')
            del image_pil, image, image_cv, image_rgb, boxes_filt, pred_phrases, masks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    del model, predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    config = dict(
        config_file=pkg_resources.resource_filename('groundingdino', 'config/GroundingDINO_SwinT_OGC.py'),
        grounded_checkpoint=str(model_dir / "groundingdino_swint_ogc.pth"),
        sam_version="vit_h",
        sam_checkpoint=str(model_dir / "sam_vit_h_4b8939.pth"),
        sam_hq_checkpoint=str(model_dir / "sam_hq_vit_h.pth"),
        use_sam_hq=True,
        image_path=r'.\testImage',
        text_prompt="Yellow tree root",
        output_dir=r'.\output_image',
        box_threshold=0.3,
        text_threshold=0.3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        create_cutout=True,
        create_transparent=True,
        processed_dir=r'.\processed_result',
        image_suffixes=IMAGE_SUFFIXES,
        recursive=True,
        batch_size=300,


        json_root=None,
        grounded_root=None,
        transparent_root=None,
        raw_root=None,
        mask_root=None,
        temp_root=None
    )
    parser = argparse.ArgumentParser(description="Ginseng extractor with folder-preserving output.")
    parser.add_argument("--input-root", default=None, help="Input root folder containing images.")
    parser.add_argument("--output-root", default=None, help="Output root for cutouts.")
    parser.add_argument("--processed-root", default=None, help="Output root for processed results.")
    parser.add_argument("--batch-size", type=int, default=None, help="Images per batch.")
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders recursively.")
    parser.add_argument("--suffixes", default=None, help="Comma-separated suffixes, e.g. .jpg,.png")
    parser.add_argument("--text-prompt", default=None, help="Text prompt for GroundingDINO.")
    parser.add_argument("--no-cutout", dest="create_cutout", action="store_false")
    parser.add_argument("--no-transparent", dest="create_transparent", action="store_false")
    parser.add_argument("--json-root", default=None, help="Output root for mask JSON files.")
    parser.add_argument("--groundsam-root", default=None, help="Output root for grounded_sam visualizations.")
    parser.add_argument("--transparent-root", default=None, help="Output root for transparent PNG cutouts.")
    parser.add_argument("--raw-root", default=None, help="Output root for raw_ image copies.")
    parser.add_argument("--mask-root", default=None, help="Output root for mask_ visualization images.")
    parser.add_argument("--temp-root", default=None, help="Output root for temp_red_* intermediate images.")
    args = parser.parse_args()

    if args.input_root:
        config['image_path'] = args.input_root
    if args.output_root:
        config['output_dir'] = args.output_root
    if args.processed_root:
        config['processed_dir'] = args.processed_root
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.recursive:
        config['recursive'] = True
    if args.suffixes:
        config['image_suffixes'] = tuple(
            s.strip().lower() for s in args.suffixes.split(',') if s.strip()
        )
    if args.text_prompt:
        config['text_prompt'] = args.text_prompt
    if args.create_cutout is False:
        config['create_cutout'] = False
    if args.create_transparent is False:
        config['create_transparent'] = False
    if args.json_root:
        config['json_root'] = args.json_root
    if args.groundsam_root:
        config['grounded_root'] = args.groundsam_root
    if args.transparent_root:
        config['transparent_root'] = args.transparent_root
    if args.raw_root:
        config['raw_root'] = args.raw_root
    if args.mask_root:
        config['mask_root'] = args.mask_root
    if args.temp_root:
        config['temp_root'] = args.temp_root

    ensure_dir(config['output_dir'])
    ensure_dir(config['processed_dir'])

    image_paths = collect_image_paths(
        config['image_path'],
        suffixes=config['image_suffixes'],
        recursive=config['recursive']
    )
    if not image_paths:
        print(f"No images found under: {config['image_path']}")
        return

    batch_size = int(config['batch_size'])
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    for i, batch_files in enumerate(iter_batches(image_paths, batch_size), start=1):
        print(f"\nProcessing batch {i}/{num_batches}: {len(batch_files)} images")
        process_batch(batch_files, config)

    postprocess_all(config['output_dir'], config['processed_dir'], config.get('temp_root'))

if __name__ == "__main__":
    main()
