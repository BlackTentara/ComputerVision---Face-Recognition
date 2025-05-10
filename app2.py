import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
from PIL import Image


# -------------------------------
# Load known faces
@st.cache_resource(show_spinner=False)
def load_dataset(dataset_path="dataset"):
    known_encodings = []
    known_names = []

    for file in os.listdir(dataset_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dataset_path, file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file)[0])
    return known_encodings, known_names

# -------------------------------
# Load reference images
@st.cache_resource(show_spinner=False)
def load_reference_images(dataset_path="dataset"):
    reference_images = {}
    for file in os.listdir(dataset_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(dataset_path, file)
            image = Image.open(img_path).convert("RGB")
            reference_images[name] = image
    return reference_images

# -------------------------------
# Face recognition logic
def recognize_faces(frame, known_encodings, known_names, threshold=0.45):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    face_names = []
    face_scores = []

    for encoding in encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        if len(distances) > 0:
            min_index = np.argmin(distances)
            min_distance = distances[min_index]
            score = 1 - min_distance
            if min_distance < threshold:
                face_names.append(known_names[min_index])
                face_scores.append(score)
            else:
                face_names.append("Unknown")
                face_scores.append(score)
        else:
            face_names.append("Unknown")
            face_scores.append(0)

    return locations, face_names, face_scores

# -------------------------------
# Drawing detection boxes
def draw_results(image, locations, names, scores):
    for (top, right, bottom, left), name, score in zip(locations, names, scores):
        label = f"{name}" if name != "Unknown" else name
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    return image

# -------------------------------
# Streamlit UI
st.title("Face Recognition App")
mode = st.radio("Pilih mode", ("Static Image", "Real-time Webcam", "Tambah Dataset"))
threshold = st.slider("Threshold Kemiripan (semakin kecil = lebih ketat)", 0.3, 0.6, 0.45, 0.01)

# Load dataset dan reference images
known_encodings, known_names = load_dataset("dataset")
reference_images = load_reference_images("dataset")

# -------------------------------
# Static Image Mode
if mode == "Static Image":
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        locations, names, scores = recognize_faces(frame_bgr, known_encodings, known_names, threshold)
        result = draw_results(frame_bgr.copy(), locations, names, scores)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Hasil Deteksi:")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)
            if names:
                st.markdown("**Nama yang terdeteksi:**")
                for name in names:
                    st.write(f"- {name}")
            else:
                st.write("Tidak ada wajah terdeteksi.")
        with col2:
            st.markdown("### Gambar Referensi:")
            shown_names = set()
            for name in names:
                if name != "Unknown" and name not in shown_names:
                    shown_names.add(name)
                    if name in reference_images:
                        st.image(reference_images[name], caption=f"Referensi: {name}", use_container_width=True)
            if not shown_names:
                st.write("Tidak ada wajah dikenali.")

# -------------------------------
# Real-time Webcam Mode
elif mode == "Real-time Webcam":
    stframe = st.empty()
    ref_col = st.sidebar

    if "run_webcam" not in st.session_state:
        st.session_state["run_webcam"] = False

    st.session_state["run_webcam"] = st.checkbox("Mulai Kamera", value=st.session_state["run_webcam"])
    run = st.session_state["run_webcam"]

    ref_col.markdown("### Wajah yang Terdeteksi:")
    ref_image_slot = ref_col.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Kamera tidak tersedia.")
        else:
            try:
                while True:
                    if not st.session_state.get("run_webcam", False):
                        break

                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Tidak dapat membaca frame dari kamera.")
                        break

                    locations, names, scores = recognize_faces(frame, known_encodings, known_names, threshold)
                    result = draw_results(frame.copy(), locations, names, scores)

                    stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                    recognized_name = None
                    for name in names:
                        if name != "Unknown":
                            recognized_name = name
                            break

                    if recognized_name and recognized_name in reference_images:
                        ref_image_slot.image(reference_images[recognized_name], caption=f"Referensi: {recognized_name}", use_container_width=True)
                    else:
                        ref_image_slot.info("Tidak ada wajah dikenali.")
            finally:
                cap.release()


# Tambah Dataset Mode
elif mode == "Tambah Dataset":
    st.markdown("## Tambahkan Wajah ke Dataset")
    st.info("Klik 'Mulai Kamera' lalu hadapkan wajah Anda ke kamera. Setelah wajah terdeteksi, isi nama dan simpan.")

    # Inisialisasi state
    if "add_dataset_started" not in st.session_state:
        st.session_state["add_dataset_started"] = False
    if "captured_face" not in st.session_state:
        st.session_state["captured_face"] = None
    if "is_reloading" not in st.session_state:
        st.session_state["is_reloading"] = False

    # Blok aksi saat reload sedang berlangsung
    if st.session_state["is_reloading"]:
        st.warning("Dataset sedang dimuat ulang. Silakan tunggu beberapa detik...")
        st.stop()

    # Tombol mulai kamera
    if st.button("Mulai Kamera"):
        st.session_state["add_dataset_started"] = True
        st.session_state["captured_face"] = None

    # Tangkap wajah dari kamera
    if st.session_state["add_dataset_started"] and st.session_state["captured_face"] is None:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Gagal membaca frame dari kamera.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)

            if locations:
                top, right, bottom, left = locations[0]
                st.session_state["captured_face"] = rgb  # Simpan seluruh frame, bukan hanya crop
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                stframe.image(frame, channels="BGR", caption="Wajah Ditemukan!", use_container_width=True)
                break
            else:
                stframe.image(frame, channels="BGR", caption="Mencari wajah...", use_container_width=True)

        cap.release()

    # Form simpan wajah
    if st.session_state["captured_face"] is not None:
        name = st.text_input("Masukkan nama untuk disimpan:")
        if name:
            if st.button("Simpan Wajah"):
                save_path = os.path.join("dataset", f"{name}.jpg")

                if os.path.exists(save_path):
                    st.error(f"Nama '{name}' sudah ada di dataset. Silakan gunakan nama lain.")
                else:
                    img = Image.fromarray(st.session_state["captured_face"])
                    
                    # Validasi encoding sebelum menyimpan
                    encodings = face_recognition.face_encodings(np.array(img))
                    if not encodings:
                        st.error("Encoding wajah gagal. Coba ulangi pengambilan gambar.")
                    else:
                        img.save(save_path)
                        st.success(f"Wajah berhasil disimpan sebagai {name}.jpg di folder dataset.")
                        

                        # Proses reload dengan spinner dan proteksi
                        st.session_state["is_reloading"] = True
                        with st.spinner("Menyimpan dan memuat ulang dataset..."):
                            load_dataset.clear()
                            load_reference_images.clear()
                            known_encodings, known_names = load_dataset("dataset")
                            reference_images = load_reference_images("dataset")
                        st.session_state["is_reloading"] = False

                        # Reset sesi
                        st.session_state["add_dataset_started"] = False
                        st.session_state["captured_face"] = None
    
