# tugas14-citra

NAMA : Ridha Muhammad Rifqi

NIM : 312210491

KELAS : TI.22.A.5


## SOURCE CODE
```
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://i.pinimg.com/564x/05/88/0d/05880df414a4426c52e32434a58c0ebb.jpg'
response = requests.get(url, stream=True)

with open('image.png', 'wb') as f:
    f.write(response.content)

img = cv2.imread('image.png')

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

u, s, v = np.linalg.svd(gray_image, full_matrices=False)

print(f'u.shape:{u.shape}, s.shape:{s.shape}, v.shape:{v.shape}')

var_explained = np.round(s*2 / np.sum(s*2), decimals=6)

print(f'Variance Explained by Top 20 singular values:\n{var_explained[0:20]}')

sns.barplot(x=list(range(1, 21)), y=var_explained[0:20], color="dodgerblue")

plt.title('Variance Explained Graph')
plt.xlabel('Singular Vector', fontsize=16)
plt.ylabel('Variance Explained', fontsize=16)
plt.tight_layout()
plt.show()

comps = [3648, 1, 5, 10, 15, 20]
plt.figure(figsize=(12, 6))

for i in range(len(comps)):
    low_rank = u[:, :comps[i]] @ np.diag(s[:comps[i]]) @ v[:comps[i], :]

    plt.subplot(2, 3, i + 1)
    plt.imshow(low_rank, cmap='gray')
    if i == 0:
        plt.title(f'Actual Image with n_components = {comps[i]}')
    else:
        plt.title(f'n_components = {comps[i]}')

plt.tight_layout()
plt.show()

```

## OUTPUT

![Screenshot 2024-06-13 112624](https://github.com/dhomuhammad/tugas14-citra/assets/130027527/40969a99-1110-40c6-9a8f-d52759590aa8)

![Screenshot 2024-06-13 112645](https://github.com/dhomuhammad/tugas14-citra/assets/130027527/7c2c3499-f802-4083-a1ca-574f07c656b5)






