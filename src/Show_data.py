from load_data import *

label_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
             4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker',
             8: 'Bag', 9: 'Ankle Boot'}

print("Sample of the Data: ")
for i in range(4):
    plt.imshow(train_data.data[i], cmap='gray')
    plt.title(label_map[train_data.targets[i].item()])
    plt.show()