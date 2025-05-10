import cv2
import face_recognition
import os
import numpy as np

def load_and_encode_faces(dataset_path="dataset"):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".jpeg") or file_name.endswith(".jpg"):
            image_path = os.path.join(dataset_path, file_name)
            image = face_recognition.load_image_file(image_path)
            name = os.path.splitext(file_name)[0]

            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)

    return known_face_encodings, known_face_names

def recognize_faces_in_frame(frame, known_face_encodings, known_face_names, recognition_threshold=0.5):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < recognition_threshold:
                name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names

def display_results(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    return frame

def resize_image_by_height(frame, height=600):
    current_height, current_width = frame.shape[:2]
    aspect_ratio = current_width / current_height
    new_height = height
    new_width = int(new_height * aspect_ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    return resized_frame

# Real-time video face recognition
def real_time_face_recognition(dataset_path="dataset"):
    known_face_encodings, known_face_names = load_and_encode_faces(dataset_path)
    print(f"Loaded {len(known_face_names)} faces from dataset.")

    video_capture = cv2.VideoCapture(0)
    window_title = "Real-Time Face Recognition"  # Single static window title
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        face_locations, face_names = recognize_faces_in_frame(
            frame, known_face_encodings, known_face_names, recognition_threshold=0.5
        )

        frame_with_results = display_results(frame, face_locations, face_names)
        cv2.imshow(window_title, resize_image_by_height(frame_with_results, height=600))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def static_image_face_recognition(image_path, dataset_path="dataset", result_folder="result"):
    known_face_encodings, known_face_names = load_and_encode_faces(dataset_path)
    print(f"Loaded {len(known_face_names)} faces from dataset.")

    frame = cv2.imread(image_path)
    face_locations, face_names = recognize_faces_in_frame(
        frame, known_face_encodings, known_face_names, recognition_threshold=0.5
    )
    frame_with_results = display_results(frame, face_locations, face_names)

    # Ensure the result folder exists
    os.makedirs(result_folder, exist_ok=True)

    for name in set(face_names):
        if name != "Unknown":
            output_file = os.path.join(result_folder, f"{name}_result.jpg")
            cv2.imwrite(output_file, frame_with_results)
            print(f"Result saved to {output_file}")

    window_title = "Recognized Faces: " + ", ".join(set(face_names)) if face_names else "No Faces Detected"
    cv2.imshow(window_title, resize_image_by_height(frame_with_results, height=600))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Static image example
    static_image_face_recognition("trial/KikoScann.jpg", "dataset")

    # Real-time video example
     
     #real_time_face_recognition("dataset")

if __name__ == "__main__":
    main()
