import matplotlib.pyplot as plt
import csv


data_path = "S:\\IOD\\Data1\\"
print("data_path = ",data_path) 

x = []
y = []
  
with open(data_path + 'Light.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        x.append(row[1])
        y.append(row[2])
  
plt.plot(x, y, color = 'g', linestyle = 'dashed',
         marker = 'o',label = "Ambient Data")
  
plt.xticks(rotation = 25)
plt.xlabel('Seconds')
plt.ylabel('Lux (Light Intensity)')
plt.title('Ambient Value ', fontsize = 20)
plt.grid()
plt.legend()
plt.show()