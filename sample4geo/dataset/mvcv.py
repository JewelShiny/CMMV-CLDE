import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
import json


class MVCVDatasetTrain(Dataset):

    def __init__(
        self,
        data_folder,
        transforms_query=None,
        transforms_reference=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        shuffle_batch_size=128,
    ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query  # ground
        self.transforms_reference = transforms_reference  # satellite

        self.df = pd.read_csv(f"{data_folder}/train.csv", header=None)

        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat"})

        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground1 = dict(zip(self.df.idx, self.df.ground1))
        self.idx2ground2 = dict(zip(self.df.idx, self.df.ground2))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground1, self.df.ground2)
        )

        self.idx2pair = dict()
        train_ids_list = list()

        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

    def __getitem__(self, index):

        idx, sat, ground1, ground2 = self.idx2pair[self.samples[index]]

        # load query -> ground image
        query_img1 = cv2.imread(f"{self.data_folder}/{ground1}")
        query_img1 = cv2.cvtColor(query_img1, cv2.COLOR_BGR2RGB)

        # load query -> ground image
        query_img2 = cv2.imread(f"{self.data_folder}/{ground2}")
        query_img2 = cv2.cvtColor(query_img2, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(f"{self.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img1 = cv2.flip(query_img1, 1)
            query_img2 = cv2.flip(query_img2, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img1 = self.transforms_query(image=query_img1)["image"]
            query_img2 = self.transforms_query(image=query_img2)["image"]

        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)["image"]

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))

            # # use roll for ground view if rotate sat view
            # c, h, w = query_img1.shape
            # shifts = - w//4 * r
            # query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img1, query_img2, reference_img, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        custom shuffle function for unique class_id sampling in batch
        """

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if (
                    idx not in idx_batch
                    and idx not in idx_epoch
                    and len(current_batch) < self.shuffle_batch_size
                ):

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    if (
                        sim_dict is not None
                        and len(current_batch) < self.shuffle_batch_size
                    ):

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(
                            near_similarity[:neighbour_split]
                        )

                        far_neighbours = copy.deepcopy(
                            near_similarity[neighbour_split:]
                        )

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch
                            if (
                                idx_near not in idx_batch
                                and idx_near not in idx_epoch
                                and idx_near
                            ):

                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print(
            "Original Length: {} - Length after Shuffle: {}".format(
                len(self.train_ids), len(self.samples)
            )
        )
        print("Break Counter:", break_counter)
        print(
            "Pairs left out of last batch to avoid creating noise:",
            len(self.train_ids) - len(self.samples),
        )
        print(
            "First Element ID: {} - Last Element ID: {}".format(
                self.samples[0], self.samples[-1]
            )
        )


class MVCVDatasetEval(Dataset):

    def __init__(
        self,
        data_folder,
        split,
        img_type,
        transforms=None,
    ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        if split == "train":
            self.df = pd.read_csv(f"{data_folder}/train.csv", header=None)
        else:
            self.df = pd.read_csv(f"{data_folder}/val.csv", header=None)

        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat"})

        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground1 = dict(zip(self.df.idx, self.df.ground1))
        self.idx2ground2 = dict(zip(self.df.idx, self.df.ground2))

        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values

        elif self.img_type == "query1":
            self.images = self.df.ground1.values
            self.label = self.df.idx.values

        elif self.img_type == "query2":
            self.images = self.df.ground2.values
            self.label = self.df.idx.values
        else:
            raise ValueError(
                "Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'"
            )

    def __getitem__(self, index):

        img = cv2.imread(f"{self.data_folder}/{self.images[index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)

class MVCVDatasetTrainSingle(Dataset):

    def __init__(
        self,
        data_folder,
        transforms_query=None,
        transforms_reference=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        multi_ground = "first",
        shuffle_batch_size=128,
    ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query  # ground
        self.transforms_reference = transforms_reference  # satellite

        self.df = pd.read_csv(f"{data_folder}/train_text.csv", header=None) 

        # 1. 重命名列
        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat", 3: "ground1_caption", 4: "ground2_caption", 5: "sat_caption"})

        # 2. 创建一个新的 DataFrame 来存储结果
        self.multi_ground = multi_ground
        rows = []

        # 3. 遍历每一行并创建两行
        for index, row in self.df.iterrows():
            if self.multi_ground=="first" or self.multi_ground=="both":
                # 第一行
                rows.append({
                    "ground": row["ground1"],
                    "sat": row["sat"],
                })
            if self.multi_ground=="second" or self.multi_ground=="both":
                # 第二行
                rows.append({
                    "ground": row["ground2"],
                    "sat": row["sat"],
                })

        # 4. 创建新的 DataFrame
        self.df = pd.DataFrame(rows)

        # self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        self.df["idx"] = self.df.index  # 使用行索引作为唯一标识符

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground)
        )

        self.idx2pair = dict()
        train_ids_list = list()

        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

    def __getitem__(self, index):

        idx, sat, ground= self.idx2pair[self.samples[index]]

        # load query -> ground image
        query_img = cv2.imread(f"{self.data_folder}/{ground}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(f"{self.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)["image"]

        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)["image"]

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))

            # # use roll for ground view if rotate sat view
            # c, h, w = query_img1.shape
            # shifts = - w//4 * r
            # query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img, reference_img,label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        custom shuffle function for unique class_id sampling in batch
        """

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if (
                    idx not in idx_batch
                    and idx not in idx_epoch
                    and len(current_batch) < self.shuffle_batch_size
                ):

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    if (
                        sim_dict is not None
                        and len(current_batch) < self.shuffle_batch_size
                    ):

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(
                            near_similarity[:neighbour_split]
                        )

                        far_neighbours = copy.deepcopy(
                            near_similarity[neighbour_split:]
                        )

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch
                            if (
                                idx_near not in idx_batch
                                and idx_near not in idx_epoch
                                and idx_near
                            ):

                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print(
            "Original Length: {} - Length after Shuffle: {}".format(
                len(self.train_ids), len(self.samples)
            )
        )
        print("Break Counter:", break_counter)
        print(
            "Pairs left out of last batch to avoid creating noise:",
            len(self.train_ids) - len(self.samples),
        )
        print(
            "First Element ID: {} - Last Element ID: {}".format(
                self.samples[0], self.samples[-1]
            )
        )


class MVCVDatasetTrainSingleRemote(Dataset):

    def __init__(
        self,
        data_folder,
        transforms_query=None,
        transforms_reference=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        multi_ground = "first",
        shuffle_batch_size=128,
    ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query  # ground
        self.transforms_reference = transforms_reference  # satellite

        self.df = pd.read_csv(f"{data_folder}/train_text.csv", header=None) 

        # 1. 重命名列
        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat", 3: "ground1_caption", 4: "ground2_caption", 5: "sat_caption"})

        # 2. 创建一个新的 DataFrame 来存储结果
        self.multi_ground = multi_ground
        rows = []

        # 3. 遍历每一行并创建两行
        for index, row in self.df.iterrows():
            if self.multi_ground=="first" or self.multi_ground=="both":
                # 第一行
                rows.append({
                    "ground": row["ground1"],
                    "sat": row["sat"],
                })
            if self.multi_ground=="second" or self.multi_ground=="both":
                # 第二行
                rows.append({
                    "ground": row["ground2"],
                    "sat": row["sat"],
                })

        # 4. 创建新的 DataFrame
        self.df = pd.DataFrame(rows)

        # self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        self.df["idx"] = self.df.index  # 使用行索引作为唯一标识符

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground)
        )

        self.idx2pair = dict()
        train_ids_list = list()

        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

    def __getitem__(self, index):

        idx, sat, ground= self.idx2pair[self.samples[index]]

        # load query -> ground image
        query_img = cv2.imread(f"{self.data_folder}/{ground}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(f"{self.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)["image"]


        if self.transforms_reference is not None:
            reference_img1 = self.transforms_reference(image=reference_img)["image"]
        if self.transforms_reference is not None:
            reference_img2 = self.transforms_reference(image=reference_img)["image"]

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2))

            # # use roll for ground view if rotate sat view
            # c, h, w = query_img1.shape
            # shifts = - w//4 * r
            # query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img, reference_img1,reference_img2,label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        custom shuffle function for unique class_id sampling in batch
        """

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if (
                    idx not in idx_batch
                    and idx not in idx_epoch
                    and len(current_batch) < self.shuffle_batch_size
                ):

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    if (
                        sim_dict is not None
                        and len(current_batch) < self.shuffle_batch_size
                    ):

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(
                            near_similarity[:neighbour_split]
                        )

                        far_neighbours = copy.deepcopy(
                            near_similarity[neighbour_split:]
                        )

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch
                            if (
                                idx_near not in idx_batch
                                and idx_near not in idx_epoch
                                and idx_near
                            ):

                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print(
            "Original Length: {} - Length after Shuffle: {}".format(
                len(self.train_ids), len(self.samples)
            )
        )
        print("Break Counter:", break_counter)
        print(
            "Pairs left out of last batch to avoid creating noise:",
            len(self.train_ids) - len(self.samples),
        )
        print(
            "First Element ID: {} - Last Element ID: {}".format(
                self.samples[0], self.samples[-1]
            )
        )

class MVCVDatasetEvalSingle(Dataset):

    def __init__(
        self,
        data_folder,
        split,
        img_type,
        transforms=None,
        multi_ground = "first",
    ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        if split == "train":
            self.df = pd.read_csv(f"{data_folder}/train_text.csv", header=None)
        else:
            self.df = pd.read_csv(f"{data_folder}/val_text.csv", header=None)

        # 1. 重命名列
        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat", 3: "ground1_caption", 4: "ground2_caption", 5: "sat_caption"})

        # 2. 创建一个新的 DataFrame 来存储结果
        self.multi_ground = multi_ground
        rows = []

        # 3. 遍历每一行并创建两行
        for index, row in self.df.iterrows():
            if self.multi_ground=="first" or self.multi_ground=="both":
                # 第一行
                rows.append({
                    "ground": row["ground1"],
                    "sat": row["sat"],
                })
            if self.multi_ground=="second" or self.multi_ground=="both":
                # 第二行
                rows.append({
                    "ground": row["ground2"],
                    "sat": row["sat"],
                })

        # 4. 创建新的 DataFrame
        self.df = pd.DataFrame(rows)


        # self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        self.df["idx"] = self.df.index  # 使用行索引作为唯一标识符

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground)
        )



        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values
        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.label = self.df.idx.values
        else:
            raise ValueError(
                "Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'"
            )

    def __getitem__(self, index):

        img = cv2.imread(f"{self.data_folder}/{self.images[index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)
    
class MVCVDatasetTrainText(Dataset):

    def __init__(
        self,
        data_folder,
        transforms_query=None,
        transforms_reference=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        text_sample = "sentence",
        multi_ground = "both",
        shuffle_batch_size=128,
    ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query  # ground
        self.transforms_reference = transforms_reference  # satellite

        self.df = pd.read_csv(f"{data_folder}/train_text.csv", header=None) 

        # 1. 重命名列
        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat", 3: "ground1_caption", 4: "ground2_caption", 5: "sat_caption"})

        # 2. 创建一个新的 DataFrame 来存储结果
        self.multi_ground = multi_ground
        rows = []

        # 3. 遍历每一行并创建两行
        for index, row in self.df.iterrows():
            if self.multi_ground=="first" or self.multi_ground=="both":
                # 第一行
                rows.append({
                    "ground": row["ground1"],
                    "sat": row["sat"],
                    "caption": row["ground1_caption"],
                    "sat_caption": row["sat_caption"]
                })
            if self.multi_ground=="second" or self.multi_ground=="both":
                # 第二行
                rows.append({
                    "ground": row["ground2"],
                    "sat": row["sat"],
                    "caption": row["ground2_caption"],
                    "sat_caption": row["sat_caption"]
                })

        # 4. 创建新的 DataFrame
        self.df = pd.DataFrame(rows)

        # self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        self.df["idx"] = self.df.index  # 使用行索引作为唯一标识符

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground, self.df.sat_caption, self.df.caption)
        )

        assert text_sample in ("sentence","paragraph","none")
        self.text_sample = text_sample

        self.idx2pair = dict()
        train_ids_list = list()

        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

    def __getitem__(self, index):

        idx, sat, ground, sat_caption, caption= self.idx2pair[self.samples[index]]

        # load query -> ground image
        query_img = cv2.imread(f"{self.data_folder}/{ground}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(f"{self.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)["image"]

        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)["image"]

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))

            # # use roll for ground view if rotate sat view
            # c, h, w = query_img1.shape
            # shifts = - w//4 * r
            # query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

        if self.text_sample == "sentence":
            # 按句子分割文本
            sentences = caption.split('.')
            # 过滤掉空句子并去除首尾空格
            sentences = [s.strip() for s in sentences if s.strip()]
            caption = random.choice(sentences)
        elif self.text_sample =="paragraph":
            # 按段落分割文本
            paragraphs = caption.strip().split('\n')  # 按两个换行符分割段落
            # 过滤掉空段落
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            caption = random.choice(paragraphs)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img, reference_img, caption,sat_caption,label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        custom shuffle function for unique class_id sampling in batch
        """

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if (
                    idx not in idx_batch
                    and idx not in idx_epoch
                    and len(current_batch) < self.shuffle_batch_size
                ):

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    if (
                        sim_dict is not None
                        and len(current_batch) < self.shuffle_batch_size
                    ):

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(
                            near_similarity[:neighbour_split]
                        )

                        far_neighbours = copy.deepcopy(
                            near_similarity[neighbour_split:]
                        )

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch
                            if (
                                idx_near not in idx_batch
                                and idx_near not in idx_epoch
                                and idx_near
                            ):

                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print(
            "Original Length: {} - Length after Shuffle: {}".format(
                len(self.train_ids), len(self.samples)
            )
        )
        print("Break Counter:", break_counter)
        print(
            "Pairs left out of last batch to avoid creating noise:",
            len(self.train_ids) - len(self.samples),
        )
        print(
            "First Element ID: {} - Last Element ID: {}".format(
                self.samples[0], self.samples[-1]
            )
        )


class MVCVDatasetEvalText(Dataset):

    def __init__(
        self,
        data_folder,
        split,
        img_type,
        transforms=None,
        text_sample = "sentence",
        multi_ground = "both",
    ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        if split == "train":
            self.df = pd.read_csv(f"{data_folder}/train_text.csv", header=None)
        else:
            self.df = pd.read_csv(f"{data_folder}/val_text.csv", header=None)

        # 1. 重命名列
        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat", 3: "ground1_caption", 4: "ground2_caption", 5: "sat_caption"})

        # 2. 创建一个新的 DataFrame 来存储结果
        self.multi_ground = multi_ground
        rows = []

        # 3. 遍历每一行并创建两行
        for index, row in self.df.iterrows():
            if self.multi_ground=="first" or self.multi_ground=="both":
                # 第一行
                rows.append({
                    "ground": row["ground1"],
                    "sat": row["sat"],
                    "caption": row["ground1_caption"],
                    "sat_caption": row["sat_caption"]
                })
            if self.multi_ground=="second" or self.multi_ground=="both":
                # 第二行
                rows.append({
                    "ground": row["ground2"],
                    "sat": row["sat"],
                    "caption": row["ground2_caption"],
                    "sat_caption": row["sat_caption"]
                })

        # 4. 创建新的 DataFrame
        self.df = pd.DataFrame(rows)


        # self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        self.df["idx"] = self.df.index  # 使用行索引作为唯一标识符

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground, self.df.sat_caption, self.df.caption)
        )



        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values

        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.captions = self.df.caption.values
            self.text_sample = text_sample
            self.label = self.df.idx.values
        else:
            raise ValueError(
                "Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'"
            )

    def __getitem__(self, index):

        img = cv2.imread(f"{self.data_folder}/{self.images[index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        label = torch.tensor(self.label[index], dtype=torch.long)

        if self.img_type =="query":
            caption = self.captions[index]
                        
            if self.text_sample == "sentence":
                # 按句子分割文本
                sentences = caption.split('.')
                # 过滤掉空句子并去除首尾空格
                sentences = [s.strip() for s in sentences if s.strip()]
                caption = random.choice(sentences)
            elif self.text_sample =="paragraph":
                # 按段落分割文本
                paragraphs = caption.strip().split('\n')  # 按两个换行符分割段落
                # 过滤掉空段落
                paragraphs = [p.strip() for p in paragraphs if p.strip()]
                caption = random.choice(paragraphs)

            return img, caption, label
        else:
            return img, label

    def __len__(self):
        return len(self.images)
    

class MVCVDatasetTrainTextLLM(Dataset):

    def __init__(
        self,
        data_folder,
        transforms_query=None,
        transforms_reference=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        multi_ground = "both",
        shuffle_batch_size=128,
    ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query  # ground
        self.transforms_reference = transforms_reference  # satellite

        self.df = pd.read_csv(f"{data_folder}/train_text_llm.csv", header=None) 

        # 1. 重命名列
        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat", 3: "ground1_caption", 4: "ground2_caption", 5: "sat_caption"})

        # 2. 创建一个新的 DataFrame 来存储结果
        self.multi_ground = multi_ground
        rows = []

        # 3. 遍历每一行并创建两行
        for index, row in self.df.iterrows():
            if self.multi_ground=="first" or self.multi_ground=="both":
                # 第一行
                rows.append({
                    "ground": row["ground1"],
                    "sat": row["sat"],
                    "caption": row["ground1_caption"],
                    "sat_caption": row["sat_caption"]
                })
            if self.multi_ground=="second" or self.multi_ground=="both":
                # 第二行
                rows.append({
                    "ground": row["ground2"],
                    "sat": row["sat"],
                    "caption": row["ground2_caption"],
                    "sat_caption": row["sat_caption"]
                })

        # 4. 创建新的 DataFrame
        self.df = pd.DataFrame(rows)

        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        # self.df["idx"] = self.df.index  # 使用行索引作为唯一标识符

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground, self.df.sat_caption, self.df.caption)
        )

        self.idx2pair = dict()
        train_ids_list = list()

        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

    def __getitem__(self, index):

        idx, sat, ground, sat_caption, caption= self.idx2pair[self.samples[index]]

        # load query -> ground image
        query_img = cv2.imread(f"{self.data_folder}/{ground}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(f"{self.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)["image"]

        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)["image"]

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))

            # # use roll for ground view if rotate sat view
            # c, h, w = query_img1.shape
            # shifts = - w//4 * r
            # query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

        caption = torch.from_numpy(np.load(f"{self.data_folder}/{caption}"))

        label = torch.tensor(idx, dtype=torch.long)

        return query_img, reference_img, caption,sat_caption,label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        custom shuffle function for unique class_id sampling in batch
        """

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if (
                    idx not in idx_batch
                    and idx not in idx_epoch
                    and len(current_batch) < self.shuffle_batch_size
                ):

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    if (
                        sim_dict is not None
                        and len(current_batch) < self.shuffle_batch_size
                    ):

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(
                            near_similarity[:neighbour_split]
                        )

                        far_neighbours = copy.deepcopy(
                            near_similarity[neighbour_split:]
                        )

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch
                            if (
                                idx_near not in idx_batch
                                and idx_near not in idx_epoch
                                and idx_near
                            ):

                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print(
            "Original Length: {} - Length after Shuffle: {}".format(
                len(self.train_ids), len(self.samples)
            )
        )
        print("Break Counter:", break_counter)
        print(
            "Pairs left out of last batch to avoid creating noise:",
            len(self.train_ids) - len(self.samples),
        )
        print(
            "First Element ID: {} - Last Element ID: {}".format(
                self.samples[0], self.samples[-1]
            )
        )


class MVCVDatasetEvalTextLLM(Dataset):

    def __init__(
        self,
        data_folder,
        split,
        img_type,
        transforms=None,
        multi_ground = "both",
    ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        if split == "train":
            self.df = pd.read_csv(f"{data_folder}/train_text_llm.csv", header=None)
        else:
            self.df = pd.read_csv(f"{data_folder}/val_text_llm.csv", header=None)

        # 1. 重命名列
        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat", 3: "ground1_caption", 4: "ground2_caption", 5: "sat_caption"})

        # 2. 创建一个新的 DataFrame 来存储结果
        self.multi_ground = multi_ground
        rows = []

        # 3. 遍历每一行并创建两行
        for index, row in self.df.iterrows():
            if self.multi_ground=="first" or self.multi_ground=="both":
                # 第一行
                rows.append({
                    "ground": row["ground1"],
                    "sat": row["sat"],
                    "caption": row["ground1_caption"],
                    "sat_caption": row["sat_caption"]
                })
            if self.multi_ground=="second" or self.multi_ground=="both":
                # 第二行
                rows.append({
                    "ground": row["ground2"],
                    "sat": row["sat"],
                    "caption": row["ground2_caption"],
                    "sat_caption": row["sat_caption"]
                })

        # 4. 创建新的 DataFrame
        self.df = pd.DataFrame(rows)


        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        # self.df["idx"] = self.df.index  # 使用行索引作为唯一标识符

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground, self.df.sat_caption, self.df.caption)
        )



        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values

        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.captions = self.df.caption.values
            self.label = self.df.idx.values
        else:
            raise ValueError(
                "Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'"
            )

    def __getitem__(self, index):

        img = cv2.imread(f"{self.data_folder}/{self.images[index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        label = torch.tensor(self.label[index], dtype=torch.long)

        if self.img_type =="query":
            caption = self.captions[index]
                        
            caption = torch.from_numpy(np.load(f"{self.data_folder}/{caption}"))

            return img, caption, label
        else:
            return img, label

    def __len__(self):
        return len(self.images)
    
class MVCVDatasetTrainComplex2(Dataset):

    def __init__(
        self,
        data_folder,
        transforms_query1=None,
        transforms_query2=None,
        transforms_reference1=None,
        transforms_reference2=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        shuffle_batch_size=128,
    ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query1 = transforms_query1  # ground
        self.transforms_query2 = transforms_query2  # ground
        self.transforms_reference1 = transforms_reference1  # satellite
        self.transforms_reference2 = transforms_reference2  # satellite

        self.df = pd.read_csv(f"{data_folder}/train.csv", header=None)

        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat"})

        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground1 = dict(zip(self.df.idx, self.df.ground1))
        self.idx2ground2 = dict(zip(self.df.idx, self.df.ground2))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground1, self.df.ground2)
        )

        self.idx2pair = dict()
        train_ids_list = list()

        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

    def __getitem__(self, index):

        idx, sat, ground1, ground2 = self.idx2pair[self.samples[index]]

        # load query -> ground image
        query_img1 = cv2.imread(f"{self.data_folder}/{ground1}")
        query_img1 = cv2.cvtColor(query_img1, cv2.COLOR_BGR2RGB)

        # load query -> ground image
        query_img2 = cv2.imread(f"{self.data_folder}/{ground2}")
        query_img2 = cv2.cvtColor(query_img2, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(f"{self.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img1 = cv2.flip(query_img1, 1)
            query_img2 = cv2.flip(query_img2, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query1 is not None:
            query_img1 = self.transforms_query1(image=query_img1)["image"]
        if self.transforms_query2 is not None:
            query_img2 = self.transforms_query2(image=query_img2)["image"]

        if self.transforms_reference1 is not None:
            reference_img1 = self.transforms_reference1(image=reference_img)["image"]
        if self.transforms_reference2 is not None:
            reference_img2 = self.transforms_reference2(image=reference_img)["image"]

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2))

            # # use roll for ground view if rotate sat view
            # c, h, w = query_img1.shape
            # shifts = - w//4 * r
            # query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img1, query_img2, reference_img1,reference_img2, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        custom shuffle function for unique class_id sampling in batch
        """

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if (
                    idx not in idx_batch
                    and idx not in idx_epoch
                    and len(current_batch) < self.shuffle_batch_size
                ):

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    if (
                        sim_dict is not None
                        and len(current_batch) < self.shuffle_batch_size
                    ):

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(
                            near_similarity[:neighbour_split]
                        )

                        far_neighbours = copy.deepcopy(
                            near_similarity[neighbour_split:]
                        )

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch
                            if (
                                idx_near not in idx_batch
                                and idx_near not in idx_epoch
                                and idx_near
                            ):

                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print(
            "Original Length: {} - Length after Shuffle: {}".format(
                len(self.train_ids), len(self.samples)
            )
        )
        print("Break Counter:", break_counter)
        print(
            "Pairs left out of last batch to avoid creating noise:",
            len(self.train_ids) - len(self.samples),
        )
        print(
            "First Element ID: {} - Last Element ID: {}".format(
                self.samples[0], self.samples[-1]
            )
        )

class MVCVDatasetTrainComplex(Dataset):

    def __init__(
        self,
        data_folder,
        transforms_query1=None,
        transforms_query2=None,
        transforms_reference1=None,
        transforms_reference2=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        shuffle_batch_size=128,
    ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query1 = transforms_query1  # ground
        self.transforms_query2 = transforms_query2  # ground
        self.transforms_reference1 = transforms_reference1  # satellite
        self.transforms_reference2 = transforms_reference2  # satellite

        self.df = pd.read_csv(f"{data_folder}/train.csv", header=None)

        self.df = self.df.rename(columns={0: "ground1", 1: "ground2", 2: "sat"})

        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground1 = dict(zip(self.df.idx, self.df.ground1))
        self.idx2ground2 = dict(zip(self.df.idx, self.df.ground2))

        self.pairs = list(
            zip(self.df.idx, self.df.sat, self.df.ground1, self.df.ground2)
        )

        self.idx2pair = dict()
        train_ids_list = list()

        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

    def __getitem__(self, index):

        idx, sat, ground1, ground2 = self.idx2pair[self.samples[index]]

        # load query -> ground image
        query_img1 = cv2.imread(f"{self.data_folder}/{ground1}")
        query_img1 = cv2.cvtColor(query_img1, cv2.COLOR_BGR2RGB)

        # load query -> ground image
        query_img2 = cv2.imread(f"{self.data_folder}/{ground2}")
        query_img2 = cv2.cvtColor(query_img2, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(f"{self.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img1 = cv2.flip(query_img1, 1)
            query_img2 = cv2.flip(query_img2, 1)
            reference_img = cv2.flip(reference_img, 1)


        # image transforms
        if self.transforms_query1 is not None:
            query_img1_aug1 = self.transforms_query1(image=query_img1)['image']

        if self.transforms_query2 is not None:
            query_img1_aug2 = self.transforms_query2(image=query_img1)['image']

        if self.transforms_query1 is not None:
            query_img2_aug1 = self.transforms_query1(image=query_img2)['image']

        if self.transforms_query2 is not None:
            query_img2_aug2 = self.transforms_query2(image=query_img2)['image']
            
        if self.transforms_reference1 is not None:
            reference_img1 = self.transforms_reference1(image=reference_img)['image']

        if self.transforms_reference2 is not None:
            reference_img2 = self.transforms_reference2(image=reference_img)['image']

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2))

            # # use roll for ground view if rotate sat view
            # c, h, w = query_img1.shape
            # shifts = - w//4 * r
            # query_img1 = torch.roll(query_img1, shifts=shifts, dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img1_aug1, query_img1_aug2,query_img2_aug1,query_img2_aug2, reference_img1,reference_img2, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        custom shuffle function for unique class_id sampling in batch
        """

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if (
                    idx not in idx_batch
                    and idx not in idx_epoch
                    and len(current_batch) < self.shuffle_batch_size
                ):

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    if (
                        sim_dict is not None
                        and len(current_batch) < self.shuffle_batch_size
                    ):

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(
                            near_similarity[:neighbour_split]
                        )

                        far_neighbours = copy.deepcopy(
                            near_similarity[neighbour_split:]
                        )

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch
                            if (
                                idx_near not in idx_batch
                                and idx_near not in idx_epoch
                                and idx_near
                            ):

                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print(
            "Original Length: {} - Length after Shuffle: {}".format(
                len(self.train_ids), len(self.samples)
            )
        )
        print("Break Counter:", break_counter)
        print(
            "Pairs left out of last batch to avoid creating noise:",
            len(self.train_ids) - len(self.samples),
        )
        print(
            "First Element ID: {} - Last Element ID: {}".format(
                self.samples[0], self.samples[-1]
            )
        )
