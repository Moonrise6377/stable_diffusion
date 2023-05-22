import os
import nltk
import torch.utils.data as data
from coco_dataset import CoCoDataset

nltk.download("punkt")


def get_loader(
    transform,
    mode="train",
    batch_size=1,
    vocab_threshold=None,
    vocab_file="./vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    num_workers=0,
    cocoapi_loc="/content"
):

    if not vocab_from_file:
        mode == "train"

    if mode == "train":
        if vocab_from_file:
            assert os.path.exists(
                vocab_file
            ), "vocab_file does not exist"
        img_folder = os.path.join(cocoapi_loc, "train2014/")
        annotations_file = os.path.join(
            cocoapi_loc, "annotations/captions_train2014.json"
        )

    elif mode == "test":
        img_folder = os.path.join(cocoapi_loc, "test2014/")
        annotations_file = os.path.join(
            cocoapi_loc, "annotations/image_info_test2014.json"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    dataset = CoCoDataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        img_folder=img_folder,
    )

    if mode == "train":
        indices = dataset.get_train_indices()
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=data.sampler.BatchSampler(
                sampler=initial_sampler, batch_size=dataset.batch_size, drop_last=False
            ),
        )
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return data_loader
