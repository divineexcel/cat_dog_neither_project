import os
import tensorflow_datasets as tfds
from PIL import Image

def prepare():
    # 1. Download & Extract Cats and Dogs from Oxford-IIIT Pet
    print("Downloading and processing Oxford-IIIT Pet dataset...")
    pet_ds = tfds.load("oxford_iiit_pet", split="train+test")
    
    cat_count = 0
    dog_count = 0
    max_images = 1000  # 1,000 per class gives a fast, reliable training balance

    for item in tfds.as_numpy(pet_ds):
        species = item['species']  # 0: Cat, 1: Dog
        img = Image.fromarray(item['image']).convert("RGB")
        
        # 80% to train, 20% to val
        split = "train" if (cat_count + dog_count) % 5 != 0 else "val"

        if species == 0 and cat_count < max_images:
            img.save(f"data/{split}/cat/cat_{cat_count}.jpg")
            cat_count += 1
        elif species == 1 and dog_count < max_images:
            img.save(f"data/{split}/dog/dog_{dog_count}.jpg")
            dog_count += 1

        if cat_count >= max_images and dog_count >= max_images:
            break

    print(f"Saved {cat_count} cats and {dog_count} dogs.")

    # 2. Download Caltech-101 for the Neither class
    print("Downloading and processing Caltech-101 for 'neither'...")
    caltech_ds, caltech_info = tfds.load("caltech101", split="train+test", with_info=True)
    class_names = caltech_info.features['label'].names

    # Exclude dogs/cats from the neither set
    excluded = {'dalmatian', 'cougar_face', 'cougar_body', 'leopard', 'cat', 'dog'}
    neither_count = 0

    for item in tfds.as_numpy(caltech_ds):
        label_name = class_names[item['label']].lower()
        if label_name in excluded:
            continue

        split = "train" if neither_count % 5 != 0 else "val"
        img = Image.fromarray(item['image']).convert("RGB")
        img.save(f"data/{split}/neither/caltech_{neither_count}.jpg")
        neither_count += 1

        if neither_count >= max_images:
            break

    print(f"Saved {neither_count} non-animal images to neither.")
    print("\nData preparation complete!")

if __name__ == "__main__":
    prepare()