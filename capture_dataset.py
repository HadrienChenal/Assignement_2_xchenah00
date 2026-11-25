import cv2, time, os

N = 100        
INTERVAL = 1    
OUT = "images"

os.makedirs(OUT, exist_ok=True)
cap = cv2.VideoCapture(0)
last = time.time() + 3   
count = 0

while count < N:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("camera", frame)
    if time.time() - last >= INTERVAL:
        cv2.imwrite(f"{OUT}/img_{count:03d}.jpg", frame)
        count += 1
        last = time.time()
        print(count, "/", N)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()