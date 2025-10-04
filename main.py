import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os


height, width = 800, 1000
image = np.ones((height, width, 3), dtype=np.uint8) * 255
output = image.copy()
points = []


def mouse_callback(event, x, y, flags, param):
    global points, output
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)


# Окно для рисования точек
cv2.namedWindow("Draw Points")
cv2.setMouseCallback("Draw Points", mouse_callback)
print("Кликай мышкой для точек. Нажми 'Enter' для завершения ввода.")


# Ввод точек
while True:
    cv2.imshow("Draw Points", output)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter
        break

cv2.destroyAllWindows()

points = np.array(points)
if len(points) == 0:
    print("Нет точек!")
    exit()


try:
    eps = int(input("Введите радиус кластера (eps, по умолчанию 50): ") or 50)
except ValueError:
    eps = 50
    print("Используется eps=50")

clustering = DBSCAN(eps=eps, min_samples=2).fit(points)
labels = clustering.labels_

# Рисуем кластеры
output = image.copy()
for x, y in points:
    cv2.circle(output, (x, y), 5, (0, 0, 255), -1)

palette = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (0, 128, 128), (128, 128, 0)
]

for k, col in zip(sorted(set(labels)), palette):
    cluster_points = points[labels == k].astype(np.float32)
    if k == -1:
        for x, y in cluster_points:
            overlay = output.copy()
            cv2.circle(overlay, (int(x), int(y)), 12, (100, 100, 100), -1)
            cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
        continue

    (cx, cy), radius = cv2.minEnclosingCircle(cluster_points)
    overlay = output.copy()
    cv2.circle(overlay, (int(cx), int(cy)), int(radius) + 3, col, -1)
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
    cv2.putText(output, str(len(cluster_points)), (int(cx) - 10, int(cy) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Отображение кластеров и сохранение
cv2.imshow("Clusters", output)
print("Нажмите 's' чтобы сохранить, 'q' чтобы выйти без сохранения.")
key = cv2.waitKey(0) & 0xFF
if key == ord("s"):
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/cluster_with_data.jpg", output)
    print("Сохранено в results/cluster_with_data.jpg")
elif key == ord("q"):
    print("Закрыто без сохранения.")

cv2.destroyAllWindows()
