import cv2

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
