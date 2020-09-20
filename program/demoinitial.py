import cv2

cap = cv2.VideoCapture(1);

ret = cap.set(3, 1920)
ret = cap.set(4, 1080)
print("FROM a_pic: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("FROM a_pic: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(True):
    ret, frame = cap.read()
    cv2.imshow('demoinitial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite('initial/demo.jpg',frame)
cap.release()
cv2.destroyAllWindows()