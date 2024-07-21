import cv2

# Test displaying an image
def display_image():
    img = cv2.imread('/storage/felix/Afstudeerproject/small_10HD/imgs/train/train_99_8_bird_view_frame_RGB.jpg')  # Replace with a valid image path
    if img is None:
        print("Error: Image not found")
        return
    cv2.imshow('Test Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test capturing video from the webcam
def display_video():
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Could not open video device")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_image()
    # display_video()  # Uncomment to test video
