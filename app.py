import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from utils import predict_multi_digit

st.set_page_config(
    page_title="El Yazısı Sayı Tanıma",
    page_icon="✍️",
    layout="wide"
)


def load_model():
    model_path = "model/mnist_cnn.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


def has_drawing(canvas_image: np.ndarray, threshold: int = 30, min_pixels: int = 120) -> bool:
    """
    Canvas üzerinde gerçekten çizim var mı kontrol eder.
    """
    if canvas_image is None:
        return False

    if canvas_image.shape[-1] == 4:
        rgb = canvas_image[:, :, :3]
    else:
        rgb = canvas_image

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    white_pixels = np.count_nonzero(gray > threshold)

    return white_pixels > min_pixels


def prepare_uploaded_image(uploaded_file):
    """
    Yüklenen görseli RGB formatında açar.
    """
    image = Image.open(uploaded_file).convert("RGB")
    return image


def prepare_canvas_image(canvas_result):
    """
    Canvas çıktısını PIL Image formatına çevirir.
    """
    if canvas_result is None or canvas_result.image_data is None:
        return None

    canvas_image = canvas_result.image_data.astype(np.uint8)

    if not has_drawing(canvas_image):
        return None

    rgb = canvas_image[:, :, :3]
    return Image.fromarray(rgb)


def draw_boxes_on_image(image, boxes, results):
    """
    Bulunan rakamların kutularını ve tahminlerini görsel üstüne çizer.
    """
    img_np = np.array(image)

    if len(img_np.shape) == 2:
        img_vis = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    else:
        img_vis = img_np.copy()

    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = str(results[i]["digit"])
        conf = results[i]["confidence"] * 100

        cv2.putText(
            img_vis,
            f"{label} ({conf:.0f}%)",
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return img_vis


def show_prediction_results(image_to_predict, model):
    number_text, results, processed_digits, boxes, thresh = predict_multi_digit(model, image_to_predict)

    if not number_text or len(results) == 0:
        st.warning("Geçerli bir rakam veya sayı algılanamadı.")
        return

    st.success(f"Algılanan sayı: {number_text}")

    avg_conf = np.mean([r["confidence"] for r in results]) * 100
    st.info(f"Ortalama güven oranı: %{avg_conf:.2f}")

    boxed_image = draw_boxes_on_image(image_to_predict, boxes, results)

    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        st.subheader("Tespit Edilen Rakam Bölgeleri")
        st.image(boxed_image, channels="RGB", use_container_width=True)

        st.subheader("Binary / Segmentasyon Görüntüsü")
        st.image(thresh, clamp=True, use_container_width=True)

    with col_b:
        st.subheader("Tek Tek Rakam Sonuçları")

        if len(processed_digits) > 0:
            digit_cols = st.columns(min(len(processed_digits), 5))

            for i, processed in enumerate(processed_digits):
                with digit_cols[i % min(len(processed_digits), 5)]:
                    st.image(processed.squeeze(), width=95, clamp=True)
                    st.caption(
                        f"Tahmin: {results[i]['digit']} | %{results[i]['confidence'] * 100:.1f}"
                    )

        detail_rows = []
        for i, result in enumerate(results, start=1):
            detail_rows.append({
                "Sıra": i,
                "Tahmin": result["digit"],
                "Güven (%)": round(result["confidence"] * 100, 2)
            })

        st.subheader("Detay Tablosu")
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

        if len(results) == 1:
            probs = results[0]["probs"]
            prob_df = pd.DataFrame({
                "Rakam": list(range(10)),
                "Olasılık": probs
            })
            st.subheader("Olasılık Dağılımı")
            st.bar_chart(prob_df.set_index("Rakam"))


def main():
    st.title("✍️ El Yazısı Rakam / Sayı Tanıma")
    st.write(
        "Tek rakam veya yan yana birden fazla rakam yazabilirsin. "
        "Sistem rakamları ayırıp soldan sağa okuyacaktır."
    )

    model = load_model()

    if model is None:
        st.error("Model bulunamadı. Önce `python train.py` komutunu çalıştır.")
        st.stop()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Canvas ile Yaz")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=8,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=500,
            height=200,
            drawing_mode="freedraw",
            key="canvas",
        )

        st.subheader("Veya Görsel Yükle")
        uploaded_file = st.file_uploader(
            "PNG / JPG / JPEG formatında bir sayı görseli yükle",
            type=["png", "jpg", "jpeg"]
        )

        predict_button = st.button("Tahmin Et", use_container_width=True)

    with right_col:
        st.subheader("Sonuç Alanı")

        if predict_button:
            image_to_predict = None

            if uploaded_file is not None:
                image_to_predict = prepare_uploaded_image(uploaded_file)
                st.image(image_to_predict, caption="Yüklenen Görsel", width=300)

            else:
                image_to_predict = prepare_canvas_image(canvas_result)

            if image_to_predict is None:
                st.warning("Lütfen önce bir rakam/sayı çiz veya görsel yükle.")
            else:
                show_prediction_results(image_to_predict, model)
        else:
            st.info("Tahmin görmek için solda bir sayı yazıp 'Tahmin Et' butonuna bas.")

    with st.expander("Kullanım İpuçları"):
        st.write("• Rakamları soldan sağa yaz.")
        st.write("• Rakamlar birbirine çok değmesin.")
        st.write("• Çok kalın çizim yapma.")
        st.write("• Tek satır halinde yazarsan daha iyi sonuç verir.")
        st.write("• Tek rakam yazarsan olasılık dağılımı da gösterilir.")


if __name__ == "__main__":
    main()