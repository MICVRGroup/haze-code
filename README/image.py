from pyecharts import Map

map2 = Map("山西地图", '山西', width=234, height=422)
city = ['太原市','大同市','朔州市', '忻州市', '阳泉市', '吕梁市',
     '晋中市', '长治市','晋城市', '临汾市','运城市' ]
values2 = [59.4375311,34.68465989,45.51373483,51.40631047,54.64195137,
           45.79318928,52.67534309,55.7963034,56.21624332,65.34860261,62.40228191]
map2.add('山西', city, values2, visual_range=[35, 66], maptype='山西',is_visualmap=True,is_label_show=False,visual_text_color='#001')
map2.render()

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio

img = np.array(Image.open('C:\\Users\\zh\\Desktop\\Q.png').convert('L'))
df = pd.DataFrame(img)
from openpyxl import load_workbook
from PIL import Image
df.to_excel('C:\\Users\\zhangmengru\\Desktop\\ditu.xlsx', index=False)
# 读取Excel文件
wb = pd.read_csv('C:\\Users\\zh\\Desktop\\ditu.csv')
data=wb.replace('1','34')
data=wb.replace('2','45')
data=wb.replace('3','51')
data=wb.replace('4','55')
data=wb.replace('5','59')
data=wb.replace('6','46')
data=wb.replace('7','56')
data=wb.replace('8','64')
data=wb.replace('9','56')
data=wb.replace('11','57')
data=wb.replace('12','53')
img = np.array(data)
imageio.imsave("C:\\Users\\zh\\Desktop\\keyi.jpg", img )
plt.figure("love")
plt.imshow(img)

