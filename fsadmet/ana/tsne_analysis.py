import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
class TSNEModule:
    def __init__(self, output_dir='/root/codes/MolFeSCue-master-2/tsne_visualizations', perplexity=30, n_iter=1000, random_state=42):

        self.output_dir = output_dir
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state

    def visualize(self, features, labels, epoch, title=None, filename=None, show_plot=True):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


        tsne = TSNE(n_components=2, perplexity=self.perplexity, n_iter=self.n_iter, random_state=self.random_state)
        features_2d = tsne.fit_transform(features)

        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colormap = plt.cm.get_cmap('tab10')  
        colors = [colormap(i / num_classes) for i in range(num_classes)]

        label_to_color = {label: color for label, color in zip(unique_labels, colors)}
        point_colors = [label_to_color[label] for label in labels]


        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=point_colors, cmap=None, s=15, alpha=0.7) 
        plt.title(title or f't-SNE visualization of the graph embeddings (Epoch {epoch})')
        plt.grid(True)  


        legend_labels = {f'Class {label}': color for label, color in label_to_color.items()}
        handles = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='', markersize=10) for color in legend_labels.values()]
        plt.legend(handles, legend_labels.keys(), loc='upper right', fontsize=10)  

        plt.xticks([]) 
        plt.yticks([]) 
        if epoch:
            filepath = os.path.join(self.output_dir, str(epoch))
            plt.savefig(filepath)
            print(f'filepath: ', filepath)
            print(f"t-SNE visualization saved to {filepath}")

        if show_plot:
            plt.draw()
            plt.pause(3)  
        
        plt.close()  

class PCAModule:
    def __init__(self, output_dir='/root/codes/MolFeSCue-master-2/pca_visualizations', n_components=2, random_state=42):

        self.output_dir = output_dir
        self.n_components = n_components
        self.random_state = random_state

    def visualize(self, features, labels, epoch, title=None, filename=None, show_plot=True):
        """
            PCA dimensionality reduction of features and visualization.
            : Param features: Feature data, a NumPy array of shape (n_samples, n_features).
            : Param labels: Label data, NumPy arrays or lists of length n_samples.
            : Param epoch: The number of epochs currently trained.
            : Param title: Image title, default to None.
            : Param filename: The file name to save the image, default is None (do not save).
            : Param show_plot: Whether to display the image, the default is True.
            : Param fixed_axis_limits: Fixed axis ranges, tuples of the shape (xmin, xmax, ymin, ymax), default to None (auto-adjust).
        """
        # fixed_axis_limits = [-80,100,-80,70]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        features_2d = pca.fit_transform(features)


        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colormap = plt.cm.get_cmap('tab10')  
        colors = [colormap(i / num_classes) for i in range(num_classes)]

        label_to_color = {label: color for label, color in zip(unique_labels, colors)}
        point_colors = [label_to_color[label] for label in labels]


        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=point_colors, cmap=None, s=5, alpha=0.7) 
        plt.title(title or f'PCA visualization of the graph embeddings (Epoch {epoch})')
        plt.grid(True)  


        legend_labels = {f'Class {label}': color for label, color in label_to_color.items()}
        handles = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='', markersize=10) for color in legend_labels.values()]
        plt.legend(handles, legend_labels.keys(), loc='upper right', fontsize=10)  

        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, format='png')
            print(f"PCA visualization saved to {filepath}")

        if show_plot:
            plt.draw()
            plt.pause(2)  
        
        plt.close()  

if __name__ == "__main__":

    example_features = np.random.rand(100, 50)  
    example_labels = np.random.randint(0, 2, size=100)  


    tsne_module = TSNEModule(output_dir='./tsne_visualizations', perplexity=30, n_iter=1000, random_state=42)


    tsne_module.visualize(example_features, example_labels, epoch=1, title="Example t-SNE Visualization", filename="example_tsne.png")