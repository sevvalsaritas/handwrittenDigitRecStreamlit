import numpy as np
import cv2
from PIL import Image


def preprocess_image_for_mnist(image):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if image is None:
        return None

    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if np.max(image) < 20:
        return None

    image = cv2.GaussianBlur(image, (5, 5), 0)

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Beyaz rakam / siyah arka plan standardı
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = 255 - thresh

    if np.count_nonzero(thresh) < 20:
        return None

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]

    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(w * (20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, int(h * (20.0 / w)))

    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    canvas = canvas.astype("float32") / 255.0
    canvas = np.expand_dims(canvas, axis=-1)

    return canvas


def predict_digit(model, image):
    processed = preprocess_image_for_mnist(image)

    if processed is None:
        return None, None, None, None

    input_img = np.expand_dims(processed, axis=0)
    preds = model.predict(input_img, verbose=0)[0]

    predicted_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return predicted_class, confidence, preds, processed


def segment_digits(image):
    """
    Görüntüdeki birden fazla rakamı bulur.
    Çıktı: soldan sağa sıralanmış digit crop listesi + bounding box listesi + binary görüntü
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # rakam beyaz, arka plan siyah standardı
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = 255 - thresh

    # küçük gürültüleri temizle
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_regions = []
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # çok küçük alanları alma
        if w < 5 or h < 10:
            continue

        # biraz padding ekle
        pad = 5
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, thresh.shape[1])
        y2 = min(y + h + pad, thresh.shape[0])

        digit_crop = thresh[y1:y2, x1:x2]
        digit_regions.append(digit_crop)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    # soldan sağa sırala
    sorted_items = sorted(zip(digit_regions, boxes), key=lambda item: item[1][0])

    if not sorted_items:
        return [], [], thresh

    digit_regions, boxes = zip(*sorted_items)
    return list(digit_regions), list(boxes), thresh


def predict_multi_digit(model, image):
    digit_regions, boxes, thresh = segment_digits(image)

    results = []
    processed_digits = []

    for digit_img in digit_regions:
        pred, conf, preds, processed = predict_digit(model, digit_img)
        if pred is not None:
            results.append({
                "digit": pred,
                "confidence": conf,
                "probs": preds
            })
            processed_digits.append(processed)

    number_text = "".join(str(r["digit"]) for r in results) if results else ""

    return number_text, results, processed_digits, boxes, thresh