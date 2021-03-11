import cv2
import scipy as sp
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos

f = 2000

# ファイルを読み込み グレースケール化
img = cv2.imread("flycapture_exper_room/210303/pattern1/seitsui/20305900-2021-03-03-205352.pgm", cv2.IMREAD_GRAYSCALE)
img = img[150:450, 500:900]
img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_plt)
plt.show()

# cv2.imshow('output_shapes.png', img)
# しきい値指定によるフィルタリング
_, threshold = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
# _, threshold = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)

# 輪郭を抽出
# _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

font = cv2.FONT_HERSHEY_DUPLEX

cv2.imwrite('flycapture_exper_room/210303/pattern1/seitsui/PNG2021-03-03-205352.png', img)
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
        # cv2.putText(img, "triangle{}".format(triangle), (x, y), font, 0.8, (250, 250, 0))

    elif len(approx) == 4:
        rectangle += 1
        # cv2.putText(img, "rectangle{}".format(rectangle), (x, y), font, 0.8, (250, 250, 0))

    elif len(approx) == 5:
        pentagon += 1
        # cv2.putText(img, "pentagon{}".format(pentagon), (x, y), font, 0.8, (0))

    elif 6 < len(approx) < 14:
        oval += 1
        # cv2.putText(img, "oval{}".format(oval), (x, y), font, 0.8, (0))

    else:
        circle += 1
        # cv2.putText(img, "circle{}".format(circle), (x, y), font, 0.8, (0))
# 結果の画像作成
lencenterlist = np.argsort(lencenterlist)[::-1]
centerlist = []
for i in range(len(lencenterlist)):
    centerlist.append(centers[lencenterlist[i]])
print(centerlist)
print(lencenterlist)
# マーカーの識別
# まずは組み合わせを作る
Conbination = []
for i in range(9):
    for j in range(i):
        for k in range(j):
            for l in range(k):
                conc = []
                conb = [i, j, k, l]
                for m in range(9):
                    if m not in conb:
                        conc.append(m)
                Conbination.append([conb, conc])

print("Conbination", Conbination)
Flist = []
for i in range(len(Conbination)):
    # F = np.linalg.norm(centerlist[Conbination[i][0][0]] - centerlist[Conbination[i][0][1]]) ** 2 + np.linalg.norm(centerlist[Conbination[i][1][0]] - centerlist[Conbination[i][1][1]]) ** 2 + np.linalg.norm(centerlist[Conbination[i][1][0]] - centerlist[Conbination[i][1][2]]) ** 2 + np.linalg.norm(centerlist[Conbination[i][1][1]] - centerlist[Conbination[i][1][2]]) ** 2
    F = 0
    for j in range(4):
        for k in range(j):
            F += np.linalg.norm(centerlist[Conbination[i][0][j]] - centerlist[Conbination[i][0][k]]) ** 2
    for j in range(5):
        for k in range(j):
            F += np.linalg.norm(centerlist[Conbination[i][1][j]] - centerlist[Conbination[i][1][k]]) ** 2
    Flist.append(F)
    print(F)

minindex = np.argmin(Flist)
minconb = Conbination[minindex]
print("minconb", minconb)

print(centerlist)
cv2.circle(img, (centerlist[minconb[0][0]][0], centerlist[minconb[0][0]][1]), 10, (200, 0, 0), thickness=-1)
cv2.circle(img, (centerlist[minconb[0][1]][0], centerlist[minconb[0][1]][1]), 10, (200, 0, 0), thickness=-1)
cv2.circle(img, (centerlist[minconb[0][2]][0], centerlist[minconb[0][2]][1]), 10, (200, 0, 0), thickness=-1)
cv2.circle(img, (centerlist[minconb[0][3]][0], centerlist[minconb[0][3]][1]), 10, (200, 0, 0), thickness=-1)
cv2.circle(img, (centerlist[minconb[1][0]][0], centerlist[minconb[1][0]][1]), 10, (0, 0, 200), thickness=-1)
cv2.circle(img, (centerlist[minconb[1][1]][0], centerlist[minconb[1][1]][1]), 10, (0, 0, 200), thickness=-1)
cv2.circle(img, (centerlist[minconb[1][2]][0], centerlist[minconb[1][2]][1]), 10, (0, 0, 200), thickness=-1)
cv2.circle(img, (centerlist[minconb[1][3]][0], centerlist[minconb[1][3]][1]), 10, (0, 0, 200), thickness=-1)
cv2.circle(img, (centerlist[minconb[1][4]][0], centerlist[minconb[1][4]][1]), 10, (0, 0, 200), thickness=-1)

#-------------------------------点の特定-----------------------------------
projection = np.array([centerlist[minconb[0][0]], centerlist[minconb[0][1]], centerlist[minconb[0][2]], centerlist[minconb[0][3]]])


#-------------------------------姿勢推定-----------------------------------
def fourpointest_rev(image, s, marker):
    f = 2000
    a0 = np.array([image[0][0], image[0][1], f])
    a1 = np.array([image[1][0], image[1][1], f])
    a2 = np.array([image[2][0], image[2][1], f])
    a3 = np.array([image[3][0], image[3][1], f])
    a0 = np.array(a0, dtype=float)
    a1 = np.array(a1, dtype=float)
    a2 = np.array(a2, dtype=float)
    a3 = np.array(a3, dtype=float)
    l01, l12, l02 = np.linalg.norm(marker[0] - marker[1]), np.linalg.norm(marker[1] - marker[2]), np.linalg.norm(marker[2] - marker[0])
    l03, l13, l23 = np.linalg.norm(marker[0] - marker[3]), np.linalg.norm(marker[1] - marker[3]), np.linalg.norm(marker[2] - marker[3])
    theta01 = np.arccos(np.inner(a0, a1) / ((a0[0]**2+a0[1]**2+a0[2]**2)**0.5 * (a1[0]**2+a1[1]**2+a1[2]**2)**0.5))
    theta12 = np.arccos(np.inner(a1, a2) / ((a2[0]**2+a2[1]**2+a2[2]**2)**0.5 * (a1[0]**2+a1[1]**2+a1[2]**2)**0.5))
    theta02 = np.arccos(np.inner(a0, a2) / ((a0[0]**2+a0[1]**2+a0[2]**2)**0.5 * (a2[0]**2+a2[1]**2+a2[2]**2)**0.5))
    theta03 = np.arccos(np.inner(a0, a3) / ((a0[0]**2+a0[1]**2+a0[2]**2)**0.5 * (a3[0]**2+a3[1]**2+a3[2]**2)**0.5))
    theta13 = np.arccos(np.inner(a1, a3) / ((a1[0]**2+a1[1]**2+a1[2]**2)**0.5 * (a3[0]**2+a3[1]**2+a3[2]**2)**0.5))
    theta23 = np.arccos(np.inner(a2, a3) / ((a2[0]**2+a2[1]**2+a2[2]**2)**0.5 * (a3[0]**2+a3[1]**2+a3[2]**2)**0.5))

    # s = sp.Symbol('s')
    r1 = l01/2/sin(theta01)
    r2 = l02/2/sin(theta02)
    r3 = l03/2/sin(theta03)
    r1 = np.array(r1, dtype=float);r2 = np.array(r2, dtype=float);r3 = np.array(r3, dtype=float)


    if s/2/r1 >= 1 or s/2/r2 >= 1 or s/2/r3 >= 1:
        F = 1000
    else:
        X1vec = 2 * r1 * (cos(theta01) * s / 2 / r1 + sin(theta01) * (4 * r1 ** 2 - s ** 2) ** 0.5 / 2 / r1) * a1 / np.linalg.norm(a1)
        X2vec = 2 * r2 * (cos(theta02) * s / 2 / r2 + sin(theta02) * (4 * r2 ** 2 - s ** 2) ** 0.5 / 2 / r2) * a2 / np.linalg.norm(a2)
        X3vec = 2 * r3 * (cos(theta03) * s / 2 / r3 + sin(theta03) * (4 * r3 ** 2 - s ** 2) ** 0.5 / 2 / r3) * a3 / np.linalg.norm(a3)
        X1vec = np.array(X1vec, dtype=float);X2vec = np.array(X2vec, dtype=float);X3vec = np.array(X3vec, dtype=float)
        l12 = np.array(l12, dtype=float);l23 = np.array(l23, dtype=float);l13 = np.array(l13, dtype=float)
        X0vec = a0 / (a0[0] ** 2 + a0[1] ** 2 + a0[2] ** 2) ** 0.5 * s

        F = (np.linalg.norm(X1vec - X2vec) - l12) ** 2 + (np.linalg.norm(X2vec - X3vec) - l23) ** 2 + (np.linalg.norm(X1vec - X3vec) - l13) ** 2

    return F, X0vec, X1vec, X2vec, X3vec


def fourpointest_lq_rev(s, image, marker):
    f = 2000
    a0 = np.array([image[0][0], image[0][1], f])
    a1 = np.array([image[1][0], image[1][1], f])
    a2 = np.array([image[2][0], image[2][1], f])
    a3 = np.array([image[3][0], image[3][1], f])
    a0 = np.array(a0, dtype=float);a1 = np.array(a1, dtype=float);a2 = np.array(a2, dtype=float);a3 = np.array(a3, dtype=float)
    l01, l12, l02 = np.linalg.norm(marker[0] - marker[1]), np.linalg.norm(marker[1] - marker[2]), np.linalg.norm(marker[2] - marker[0])
    l03, l13, l23 = np.linalg.norm(marker[0] - marker[3]), np.linalg.norm(marker[1] - marker[3]), np.linalg.norm(marker[2] - marker[3])
    theta01 = np.arccos(np.inner(a0, a1) / ((a0[0]**2+a0[1]**2+a0[2]**2)**0.5 * (a1[0]**2+a1[1]**2+a1[2]**2)**0.5))
    theta12 = np.arccos(np.inner(a1, a2) / ((a2[0]**2+a2[1]**2+a2[2]**2)**0.5 * (a1[0]**2+a1[1]**2+a1[2]**2)**0.5))
    theta02 = np.arccos(np.inner(a0, a2) / ((a0[0]**2+a0[1]**2+a0[2]**2)**0.5 * (a2[0]**2+a2[1]**2+a2[2]**2)**0.5))
    theta03 = np.arccos(np.inner(a0, a3) / ((a0[0]**2+a0[1]**2+a0[2]**2)**0.5 * (a3[0]**2+a3[1]**2+a3[2]**2)**0.5))
    theta13 = np.arccos(np.inner(a1, a3) / ((a1[0]**2+a1[1]**2+a1[2]**2)**0.5 * (a3[0]**2+a3[1]**2+a3[2]**2)**0.5))
    theta23 = np.arccos(np.inner(a2, a3) / ((a2[0]**2+a2[1]**2+a2[2]**2)**0.5 * (a3[0]**2+a3[1]**2+a3[2]**2)**0.5))

    # s = sp.Symbol('s')
    r1 = l01/2/sin(theta01)
    r2 = l02/2/sin(theta02)
    r3 = l03/2/sin(theta03)
    r1 = np.array(r1, dtype=float); r2 = np.array(r2, dtype=float); r3 = np.array(r3, dtype=float)

    if s/2/r1 >= 1 or s/2/r2 >= 1 or s/2/r3 >= 1:
        F = 1000
    else:
        X1vec = 2 * r1 * (cos(theta01)*s/2/r1 + sin(theta01)*(4*r1**2 - s**2)**0.5/2/r1) * a1 / np.linalg.norm(a1)
        X2vec = 2 * r2 * (cos(theta02)*s/2/r2 + sin(theta02)*(4*r2**2 - s**2)**0.5/2/r2) * a2 / np.linalg.norm(a2)
        X3vec = 2 * r3 * (cos(theta03)*s/2/r3 + sin(theta03)*(4*r3**2 - s**2)**0.5/2/r3) * a3 / np.linalg.norm(a3)
        X1vec = np.array(X1vec, dtype=float);X2vec = np.array(X2vec, dtype=float);X3vec = np.array(X3vec, dtype=float)
        l12 = np.array(l12, dtype=float);l23 = np.array(l23, dtype=float);l13 = np.array(l13, dtype=float)
        X0vec = a0 / (a0[0] ** 2 + a0[1] ** 2 + a0[2] ** 2) ** 0.5 * s

        F = (np.linalg.norm(X1vec - X2vec) - l12) ** 2 + (np.linalg.norm(X2vec - X3vec) - l23) ** 2 + (np.linalg.norm(X1vec - X3vec) - l13) ** 2
    return F


marker = np.array([[0.0, 3.0, 0.0], [-4.5, 1.5, 0], [4.5, 1.5, 0], [0.0, 8.1, 0.0]])*0.01
res = optimize.least_squares(fourpointest_lq_rev, 0.1, args=[projection, marker])
result = fourpointest_rev(projection, res.x[0], marker)
print(result)

# 図形の数の結果
print('Number of triangle = ', triangle)
print('Number of rectangle = ', rectangle)
print('Number of pentagon = ', pentagon)
print('Number of circle = ', circle)
print('Number of oval = ', oval)
img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_plt)
plt.show()
