import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset(opt)
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
        dataset.initialize(opt)
    elif opt.dataset_mode == 'Gopro':
        from data.GoProdataset import GoProDataset
        dataset = GoProDataset("train_blur_images.txt", "train_sharp_images.txt", "gt_softedge_images.txt")
    elif opt.dataset_mode == 'Gopro_test':
        from data.GoProdataset import GoProDataset_test
        # print("Gopro test")
        dataset = GoProDataset_test("test_blur_images.txt", "test_sharp_images.txt")#Ordered_dir
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    # print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, opt):
        super(CustomDatasetDataLoader,self).initialize(opt)
        print("Opt.nThreads = ", opt.nThreads)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            pin_memory=True
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
