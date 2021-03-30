import cv2
import numpy as np
import matplotlib.pyplot as plt

# ファイルを読み込み グレースケール化
img = cv2.imread("flycapture_exper_room/210303/pattern2/seitsui/20305900-2021-03-03-204831.pgm", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("flycapture_exper_room/210303/pattern2/naname/20305900-2021-03-03-204707.pgm", cv2.IMREAD_GRAYSCALE)
# img = img[150:450, 500:900]
img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_plt)
plt.show()

# cv2.imshow('output_shapes.png', img)
# しきい値指定によるフィルタリング
_, threshold = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
# _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# 輪郭を抽出
# _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
approx_list = []
for cnt in contours:
    print(cnt.T)
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    approx_list.append(approx)
    # cv2.drawContours(img, [approx], 0, (0), 2)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    cx = int(np.mean(cnt.T[0]))
    cy = int(np.mean(cnt.T[1]))
    print([cnt.T[0], y])
    centers.append(np.array([cx, cy]))
    lencenterlist.append(len(cnt.T[0][0]))

lencenterlist = np.argsort(lencenterlist)[::-1]
imagelist = []
for cnt in range(2):
    approx = approx_list[lencenterlist[cnt]]
    print("approx", approx)
    # cv2.drawContours(img, [approx], 0, (0, 0, 250), 2)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 3:
        triangle += 1
        cv2.putText(img, "triangle{}".format(triangle), (x, y), font, 0.8, (250, 250, 0))

    elif len(approx) == 4:
        rectangle += 1
        # cv2.putText(img, "rectangle{}".format(rectangle), (x, y), font, 0.8, (250, 250, 0))

    elif len(approx) == 5:
        pentagon += 1
        # cv2.putText(img, "pentagon{}".format(pentagon), (x, y), font, 0.8, (250, 250, 0))

    elif 6 < len(approx) < 14:
        oval += 1
        # cv2.putText(img, "oval{}".format(oval), (x, y), font, 0.8, (0))

    else:
        circle += 1
        # cv2.putText(img, "circle{}".format(circle), (x, y), font, 0.8, (250, 250, 0))
#-----------------------変なところにある頂点をカットする。すでにopencvで隣り合う同士は並べられているので、それが小さすぎるものを省く------------------------------------
Kado1 = approx_list[lencenterlist[0]]
Kado2 = approx_list[lencenterlist[1]]
newKado1, newKado2 = [], []
length1 = 0
length2 = 0
length1list = []
length2list = []
for i in range(len(Kado1)):
    if i == 0:
        length1 += np.linalg.norm(Kado1[1] - Kado1[0]) + np.linalg.norm(Kado1[-1] - Kado1[0])
        length1list.append([np.linalg.norm(Kado1[1] - Kado1[0]), np.linalg.norm(Kado1[-1] - Kado1[0])])
    elif i == int(len(Kado1)) - 1:
        length1 += np.linalg.norm(Kado1[-1] - Kado1[-2]) + np.linalg.norm(Kado1[-1] - Kado1[0])
        length1list.append([np.linalg.norm(Kado1[-1] - Kado1[-2]), np.linalg.norm(Kado1[-1] - Kado1[0])])
    else:
        length1 += np.linalg.norm(Kado1[i] - Kado1[i-1]) + np.linalg.norm(Kado1[i] - Kado1[i+1])
        length1list.append([np.linalg.norm(Kado1[i] - Kado1[i-1]), np.linalg.norm(Kado1[i] - Kado1[i+1])])
normlen1 = length1/np.linalg.norm(int(len(Kado1)))
for i in range(len(Kado1)):
    if length1list[i][0] > normlen1/5:#しきい値は試行錯誤しているので、ロバストとは言い難い
        newKado1.append(Kado1[i])

for i in range(len(Kado2)):
    if i == 0:
        length2 += np.linalg.norm(Kado2[0] - Kado2[-1]) + np.linalg.norm(Kado2[1] - Kado2[0])
        length2list.append([np.linalg.norm(Kado2[0] - Kado2[-1]), np.linalg.norm(Kado2[1] - Kado2[0])])
    elif i == int(len(Kado2)) - 1:
        length2 += np.linalg.norm(Kado2[-1] - Kado2[-2]) + np.linalg.norm(Kado2[-1] - Kado2[0])
        length2list.append([np.linalg.norm(Kado2[-1] - Kado2[-2]), np.linalg.norm(Kado2[-1] - Kado2[0])])
    else:
        length2 += np.linalg.norm(Kado2[i] - Kado2[i-1]) + np.linalg.norm(Kado2[i] - Kado2[i+1])
        length2list.append([np.linalg.norm(Kado2[i] - Kado2[i-1]), np.linalg.norm(Kado2[i] - Kado2[i+1])])
normlen2 = length2/np.linalg.norm(int(len(Kado2)))
print("normlen2", normlen2)
for i in range(len(Kado2)):
    print("normlen", length2list[i])
    if length2list[i][0] > normlen2/3:
        newKado2.append(Kado2[i])

KadoCount1 = int(len(newKado1))#頂点の個数
KadoCount2 = int(len(newKado2))#頂点の個数
# KadoCount1 = int(len(approx_list[lencenterlist[0]]))
# KadoCount2 = int(len(approx_list[lencenterlist[0]]))
print("kadolist", [KadoCount1, KadoCount2])
if KadoCount1 > KadoCount2:
    cv2.drawContours(img, [approx_list[lencenterlist[0]]], 0, (0, 0, 250), 2)
    cv2.drawContours(img, [approx_list[lencenterlist[1]]], 0, (250, 0, 0), 2)
    # cv2.circle(img, (centers[lencenterlist[0]][0], centers[lencenterlist[0]][1]), 10, (0, 0, 200), thickness=-1)
    cv2.putText(img, "A", (centers[lencenterlist[0]][0], centers[lencenterlist[0]][1]), font, 0.6, (0, 0, 200))
    # cv2.circle(img, (centers[lencenterlist[1]][0], centers[lencenterlist[1]][1]), 10, (200, 0, 0), thickness=-1)
    cv2.putText(img, "B", (centers[lencenterlist[1]][0], centers[lencenterlist[1]][1]), font, 0.6, (200, 0, 0))
else:
    cv2.drawContours(img, [approx_list[lencenterlist[0]]], 0, (250, 0, 0), 2)
    cv2.drawContours(img, [approx_list[lencenterlist[1]]], 0, (0, 0, 250), 2)
    # cv2.circle(img, (centers[lencenterlist[0]][0], centers[lencenterlist[0]][1]), 10, (200, 0, 0), thickness=-1)
    cv2.putText(img, "B", (centers[lencenterlist[0]][0], centers[lencenterlist[0]][1]), font, 0.6, (200, 0, 0))
    # cv2.circle(img, (centers[lencenterlist[1]][0], centers[lencenterlist[1]][1]), 10, (0, 0, 200), thickness=-1)
    cv2.putText(img, "A", (centers[lencenterlist[1]][0], centers[lencenterlist[1]][1]), font, 0.6, (0, 0, 200))
# 結果の画像作成
cv2.imwrite('flycapture_exper_room/210303/pattern2/seitsui/PNG2021-03-03-204831.png', img)


# 図形の数の結果
print('Number of triangle = ', triangle)
print('Number of rectangle = ', rectangle)
print('Number of pentagon = ', pentagon)
print('Number of circle = ', circle)
print('Number of oval = ', oval)
img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_plt)
plt.show()
