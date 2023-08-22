import matplotlib.pyplot as plt

losses=[0.5,0.3,0.2,0.15,0.1,0.08,0.06,0.05]

plt.plot(losses='total loss')
plt.plot(losses,'r')
plt.xlabel('training steps')
plt.ylabel('loss')
plt.legend()
plt.show()