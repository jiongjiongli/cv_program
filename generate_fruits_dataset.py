import random
from pathlib import Path
import shutil


def generate_dataset(data_dir_path, dataset_root_dir_path):
    train_percent = 0.8
    val_percent = 0.1
    # test_percent = 0.1
    dataset_dir_names = ['training_set', 'val_set', 'test_set']

    data_dir_path = Path(data_dir_path)
    dataset_root_dir_path = Path(dataset_root_dir_path)

    image_file_paths = list(data_dir_path.rglob('*.*'))

    for image_file_path in image_file_paths:
         assert image_file_path.suffix in ['.jpeg', '.jpg', '.png'], r'Unknown suffix: {}'.format(image_file_path)

    print('num_images:', len(image_file_paths))

    if dataset_root_dir_path.exists():
        shutil.rmtree(dataset_root_dir_path.as_posix())

    category_dir_paths = [child_path for child_path in data_dir_path.iterdir() if child_path.is_dir()]
    class_names = [category_dir_path.name for category_dir_path in category_dir_paths]
    num_classes = len(class_names)

    print('num_classes:', num_classes, class_names)

    for category_dir_path in category_dir_paths:
        category_name = category_dir_path.name
        image_file_paths = list(category_dir_path.rglob('*.*'))

        for image_file_path in image_file_paths:
            assert image_file_path.suffix in ['.jpeg', '.jpg', '.png'], r'Unknown suffix: {}'.format(image_file_path)

        random.shuffle(image_file_paths)

        num_images = len(image_file_paths)
        # print('num_images:', num_images)

        num_train_images = int(num_images * train_percent)
        num_val_images = int(num_images * val_percent)
        num_test_images = num_images - (num_train_images + num_val_images)

        # print('Expected:')
        # print('num_train_images:', num_train_images,
        #       'num_val_images:', num_val_images,
        #       'num_test_images:', num_test_images)

        train_image_file_paths = image_file_paths[:num_train_images]
        val_image_file_paths   = image_file_paths[num_train_images:-num_test_images]
        test_image_file_paths  = image_file_paths[-num_test_images:]

        dataset_file_paths_list = [train_image_file_paths, val_image_file_paths, test_image_file_paths]
        for dataset_dir_name, dataset_file_paths in zip(dataset_dir_names, dataset_file_paths_list):
            dataset_dir_path = dataset_root_dir_path / dataset_dir_name / category_name
            dataset_dir_path.mkdir(parents=True, exist_ok=True)

            for dataset_file_path in dataset_file_paths:
                dest_file_path = dataset_dir_path / dataset_file_path.name
                shutil.copyfile(dataset_file_path.as_posix(), dest_file_path.as_posix())

    for dataset_dir_name in dataset_dir_names:
        sub_dir_path = dataset_root_dir_path / dataset_dir_name
        sub_category_paths = [child_path for child_path in sub_dir_path.iterdir() if child_path.is_dir()]
        assert len(sub_category_paths) == num_classes, len(sub_category_paths)

        file_paths = list(sub_dir_path.rglob('*.*'))

        for image_file_path in file_paths:
            assert image_file_path.suffix in ['.jpeg', '.jpg', '.png'], r'Unknown suffix: {}'.format(image_file_path)

        print(r'num_images for {}:'.format(dataset_dir_name), len(file_paths))

    print('Completed!')


def main():
    data_dir_path = r'data/fruit30_train'
    dataset_root_dir_path = r'data/fruit30_dataset'
    generate_dataset(data_dir_path, dataset_root_dir_path)


if __name__ == '__main__':
    main()
