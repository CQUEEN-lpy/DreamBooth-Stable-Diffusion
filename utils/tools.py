import matplotlib.pyplot as plt
def test_generated_imgs(item):
    fig, axs = plt.subplots(1, 8, figsize=(16, 4))
    for i, image in enumerate(item):
        axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
        axs[i].set_axis_off()
    plt.show()