import seaborn as sns
import matplotlib.pyplot as plt

def plot_accuracy(list1, list2):
    # Set seaborn context to "poster" for larger, more readable plots
    sns.set_context("poster")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))  # Adjust the size as needed

    # Plotting with Seaborn
    sns.lineplot(x=range(len(list1)), y=list1, label='DINO', color='red', ax=ax)
    sns.lineplot(x=range(len(list2)), y=list2, label='DINO + TORE', color='green', ax=ax)

    # Adding labels and title with LaTeX syntax for kappa
    ax.set_xlabel(r'$\kappa$')  # Increase font size to 14
    ax.set_ylabel('$k$-NN accuracy')  # Increase font size to 14
    ax.set_title('ImageNet-1k')  # Increase font size to 16

    # Adding a legend
    ax.legend()

    # Save the plot as PNG
    plt.savefig('knn_plot.png')

    # Display the plot
    plt.show()

def main():
    # Your data
    dino_tore = [78.072, 77.308, 77.21, 77.062, 76.934, 76.896, 76.608, 76.338, 75.86, 75.476, 75.07, 74.324, 72.474]
    dino = [78.512, 76.288, 74.598, 73.238, 71.2, 70.812, 69.372, 69.41, 69.04, 68.32, 66.816, 63.518, 57.928]

    # Display the plot using Seaborn
    plot_accuracy(dino, dino_tore)

if __name__ == "__main__":
    main()