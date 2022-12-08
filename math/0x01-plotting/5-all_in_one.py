#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
fig = plt.figure()
fig.suptitle("All in One")

plt.subplot(321)
plt.plot(y0, 'r-')
plt.xlim([0, 10])
plt.ylim([0, 1000])


plt.subplot(322)
plt.xlabel("Height (in)", size='x-small')
plt.ylabel("Weight (lbs)", size='x-small')
plt.title("Men's Height vs Weight", size='x-small')
plt.scatter(x1, y1, color='magenta')


plt.subplot(323)
plt.title("Exponential Decay of C-14", size='x-small')
plt.xlabel("Time (years)", size='x-small')
plt.ylabel("Fraction Remaining", size='x-small')
plt.xlim([0, 28650])
plt.yscale("log")
plt.plot(x2, y2, c='blue')


plt.subplot(324)
plt.xlabel("Time (years)", size='x-small')
plt.ylabel("Fraction Remaining", size='x-small')
plt.title("Exponential Decay of Radioactive Elements", size='x-small')
plt.xlim([0, 20000])
plt.ylim([0, 1])
plt.plot(x3, y31, 'r--', label='C-14')
plt.plot(x3, y32, 'g', label='Ra-226')
plt.legend(loc='upper right')

plt.subplot(313)
plt.hist(student_grades, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], edgecolor='black')
plt.xlabel("Grades", size='x-small')
plt.ylabel("Number of Students", size='x-small')
plt.title("Project A", size='x-small')
plt.xlim([0, 100])
plt.xticks(np.arange(0, 101, 10))
plt.ylim([0, 30])

plt.tight_layout()
plt.show()
