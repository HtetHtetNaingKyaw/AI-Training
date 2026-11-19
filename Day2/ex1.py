import numpy as np

scores  = np.array([45, 60, 72, 88, 95, 67])

print(scores)
print("Shape:", scores.shape)
print("Mean:",np.mean(scores))
print("Median:",np.median(scores))
print("Highest score:",np.max(scores))
print("Lowest score:",np.min(scores))

bonus = scores + 5
print("Bonus scores:", bonus)

data = np.array([
    [2, 60],
    [4, 75],
    [6, 90]
])