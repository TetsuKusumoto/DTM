import cv2
import numpy as np
import matplotlib.pyplot as plt

# ファイルを読み込み グレースケール化
# img = cv2.imread("shapes.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("triangle_test3.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("flycapture_test/210209success/20305900-2021-02-09-203050.pgm", cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(10, 10))
# img2 = img[:,:,::-1]
# plt.xticks([]), plt.yticks([])
img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_plt)
plt.show()

# cv2.imshow('output_shapes.png', img)
# しきい値指定によるフィルタリング
# _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
_, threshold = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)

# 輪郭を抽出
# _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

font = cv2.FONT_HERSHEY_DUPLEX

# 図形の数の変数
triangle = 0
rectangle = 0
pentagon = 0
oval = 0
circle = 0

centers = []
lencenterlist = []
# 図形の設定
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    print(cnt.T)
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (0), 2)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    cx = int(np.mean(cnt.T[0]))
    cy = int(np.mean(cnt.T[1]))
    print([cnt.T[0], y])
    centers.append(np.array([cx, cy]))
    lencenterlist.append(len(cnt.T[0][0]))

    if len(approx) == 3:
        triangle += 1
        cv2.putText(img, "triangle{}".format(triangle), (x, y), font, 0.8, (250, 250, 0))

    elif len(approx) == 4:
        rectangle += 1
        cv2.putText(img, "rectangle{}".format(rectangle), (x, y), font, 0.8, (250, 250, 0))

    elif len(approx) == 5:
        pentagon += 1
        cv2.putText(img, "pentagon{}".format(pentagon), (x, y), font, 0.8, (0))

    elif 6 < len(approx) < 14:
        oval += 1
        cv2.putText(img, "oval{}".format(oval), (x, y), font, 0.8, (0))

    else:
        circle += 1
        cv2.putText(img, "circle{}".format(circle), (x, y), font, 0.8, (0))
# 結果の画像作成
cv2.imwrite('output_shapes.png', img)
lencenterlist = np.argsort(lencenterlist)[::-1]
centerlist = []
for i in range(len(lencenterlist)):
    centerlist.append(centers[lencenterlist[i]])
print(centerlist)
print(lencenterlist)
# マーカーの識別
# まずは組み合わせを作る
# 本当は領域の大きい順に並び変えるべき
Conbination = []
for i in range(5):
    for j in range(i):
        conc = []
        conb = [i, j]
        for k in range(5):
            if k not in conb:
                conc.append(k)
        Conbination.append([conb, conc])

print(Conbination)
Flist = []
for i in range(len(Conbination)):
    F = np.linalg.norm(centerlist[Conbination[i][0][0]] - centerlist[Conbination[i][0][1]]) ** 2 + np.linalg.norm(centerlist[Conbination[i][1][0]] - centerlist[Conbination[i][1][1]]) ** 2 + np.linalg.norm(centerlist[Conbination[i][1][0]] - centerlist[Conbination[i][1][2]]) ** 2 + np.linalg.norm(centerlist[Conbination[i][1][1]] - centerlist[Conbination[i][1][2]]) ** 2
    Flist.append(F)
    print(F)

minindex = np.argmin(Flist)
minconb = Conbination[minindex]
print(minconb)

print(centerlist)
cv2.circle(img, (centerlist[minconb[0][0]][0], centerlist[minconb[0][0]][1]), 10, (200, 0, 0), thickness=-1)
cv2.circle(img, (centerlist[minconb[0][1]][0], centerlist[minconb[0][1]][1]), 10, (200, 0, 0), thickness=-1)
cv2.circle(img, (centerlist[minconb[1][0]][0], centerlist[minconb[1][0]][1]), 10, (0, 0, 200), thickness=-1)
cv2.circle(img, (centerlist[minconb[1][1]][0], centerlist[minconb[1][1]][1]), 10, (0, 0, 200), thickness=-1)
cv2.circle(img, (centerlist[minconb[1][2]][0], centerlist[minconb[1][2]][1]), 10, (0, 0, 200), thickness=-1)

# 図形の数の結果
print('Number of triangle = ', triangle)
print('Number of rectangle = ', rectangle)
print('Number of pentagon = ', pentagon)
print('Number of circle = ', circle)
print('Number of oval = ', oval)
img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_plt)
plt.show()
