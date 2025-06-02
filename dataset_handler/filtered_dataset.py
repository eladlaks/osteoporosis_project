from dataset_handler.dataset import ImageDataset

class FilteredImageDataset(ImageDataset):
    def __init__(self, root_dir, selected_paths_set, transform=None):
        super().__init__(root_dir, transform)
        filtered_image_paths = []
        filtered_labels = []

        for path, label in zip(self.image_paths, self.labels):
            if path in selected_paths_set:
                filtered_image_paths.append(path)
                filtered_labels.append(label)

        self.image_paths = filtered_image_paths
        self.labels = filtered_labels