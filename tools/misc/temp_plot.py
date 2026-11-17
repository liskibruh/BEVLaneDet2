import matplotlib.pyplot as plt


x = [1, 2, 3, 4, 5]
y = [elem**2 for elem in x]

plt.figure(figsize=(25, 17))

plt.plot(x, y, label='mean_grad_norm_vs_iters')
plt.title("grad_norm_vs_iters")
plt.xlabel("training iters")
plt.ylabel("mean_grad_norm")
plt.legend()
plt.grid(True)
plt.show()