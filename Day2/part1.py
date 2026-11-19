import numpy as np
scores = np.array([65, 70, 85, 90, 75])
print(scores)
print("Shape:", scores.shape)
print("Dimensions:", scores.ndim)
print("Data type:", scores.dtype)

import numpy as np
data = np.array([
    [2, 60],
    [4, 75],
    [6, 90]
])
print(data)
print("Shape:", data.shape)
print("First student:", data[0])
print("Study hours:", data[:, 0])

import numpy as np
scores = np.array([65, 70, 85, 90, 75])
print("Total:", np.sum(scores))
print("Mean:", np.mean(scores))
print("Maximum:", np.max(scores))
print("Minimum:", np.min(scores))
print("Standard deviation:", np.std(scores))


import numpy as np
hours = np.array([1, 2, 3, 4, 5])
scores = hours * 10 + 30
print(scores)