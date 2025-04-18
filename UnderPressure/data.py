# Scripts from UnderPressure
import anim
import util

# Python
import csv
import re
from pathlib import Path

# Misc
"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

import torch

# Modified Data location for GroundLink
# ROOT = Path(__file__).parent / "../GRF/ProcessedData"
ROOT = Path(__file__).parent / "./ProcessedData/ProcessedData"

# Modified joint list for GroundLink
TOPOLOGY = anim.Topology([
    ('pelvis', []),
    ('left_hip', ['pelvis']),
    ('right_hip', ['pelvis']),
    ('spine1', ['pelvis']),
    ('left_knee', ['left_hip']),
    ('right_knee', ['right_hip']),
    ('spine2', ['spine1']),
    ('left_ankle', ['left_knee']),
    ('right_ankle', ['right_knee']),
    ('spine3', ['spine2']),
    ('left_foot', ['left_ankle']),
    ('right_foot', ['right_ankle']),
    ('neck', ['spine3']),
    ('left_collar', ['spine3']),
    ('right_collar', ['spine3']),
    ('head', ['neck']),
    ('left_shoulder', ['left_collar']),
    ('right_shoulder', ['right_collar']),
    ('left_elbow', ['left_shoulder']),
    ('right_elbow', ['right_shoulder']),
    ('left_wrist', ['left_elbow']),
    ('right_wrist', ['right_elbow']),
    ('hand', ['right_wrist']),
    # ('shape1', []),
    # ('shape2', []),
    # ('shape3', []),
    # ('shape4', []),
    # ('shape5', []),
    # ('shape6', []),
    # ('shape7', []),
    # ('shape8', []),
    # ('shape9', []),
    # ('shape10', []),
    # ('shape11', []),
    # ('shape12', []),
    # ('shape13', []),
    # ('shape14', []),
    # ('shape15', []),
    # ('shape16', []),
])
FRAMERATE = 250


class Subject:
    @classmethod
    def subjects_file(cls):
        return ROOT / "subjects.csv"

    def __init__(self, id: str, gender: str, age: int):
        self._id = str(id)
        self._gender = str(gender)
        self._age = int(age)
        # self._height = int(height)
        # self._shoe_length = int(shoe_length)
        # self._arm_length = int(arm_length)
        # self._weight = int(weight)
        # self._insoles_size = int(insoles_size)

    @classmethod
    def from_id(cls, id):
        if not hasattr(cls, "_SUBJECTS"):
            cls.all()  # parse subject
        return cls._SUBJECTS.get(id, None)

    @classmethod
    def from_dict(cls, input):
        # return cls(input["Id"], input["Gender"], input["Age"], input["Height"], input["Shoe Length"], input["Arm Length"], input["Weight"], input["Insoles Size"])
        return cls(input["Id"], input["Gender"], input["Age"])

    @classmethod
    def parse(cls, input):
        if isinstance(input, str):
            return cls.from_id(input)
        elif isinstance(input, Subject):
            return input
        elif isinstance(input, dict):
            return cls.from_dict(input)
        else:
            return None

    @classmethod
    def all(cls):
        if not hasattr(cls, "_SUBJECTS"):
            cls._SUBJECTS = {}
            with open(cls.subjects_file(), newline="") as csvfile:
                for row in csv.DictReader(csvfile, delimiter=";"):
                    cls._SUBJECTS[row["Id"]] = cls.from_dict(row)
        return list(cls._SUBJECTS.values())

    @classmethod
    def train(cls):
        return [cls.parse("S{}".format(i)) for i in range(1, 7)]
        # return [cls.parse("S{}".format(i)) for i in range(1, 8)]

    @classmethod
    def test(cls):
        # return [cls.parse("S{}".format(i)) for i in range(8, 11)]
        return [cls.parse("S{}".format(i)) for i in range(7, 8)]

    def __repr__(self):
        return self.id

    @property
    def id(self) -> str:
        return self._id

    @property
    def gender(self) -> str:
        return self._gender

    @property
    def age(self) -> str:
        return self._age

    @property
    def height(self) -> str:
        return self._height

    @property
    def shoe_length(self) -> str:
        return self._shoe_length

    @property
    def arm_length(self) -> str:
        return self._arm_length

    @property
    def weight(self) -> str:
        return self._weight

    @property
    def insoles_size(self) -> str:
        return self._insoles_size

    def dir(self):
        return ROOT / self.id

    def insoles_txt_dir(self):
        return self.dir() / "moticon.txt"

    def insoles_txt_files(self):
        return self.insoles_txt_dir().iterdir() if self.insoles_txt_dir().is_dir() else []

    def insoles_txt_file(self, sequence: str):
        return self.insoles_txt_dir() / (sequence + ".txt")

    def mocap_mvnx_dir(self):
        return self.dir() / "xsens.mvnx"

    def mocap_mvnx_files(self):
        return self.mocap_mvnx_dir().iterdir() if self.mocap_mvnx_dir().is_dir() else []

    def mocap_mvnx_file(self, sequence: str):
        return self.mocap_mvnx_dir() / (sequence + ".mvnx")

    def sync_csv_dir(self):
        return self.dir() / "sync.csv"

    def sync_csv_files(self):
        return self.sync_csv_dir().iterdir() if self.sync_csv_dir().is_dir() else []

    def sync_csv_file(self, sequence: str):
        return self.sync_csv_dir() / (sequence + ".csv")

    def preprocessed_dir(self):
        return self.dir() / "preprocessed"

    def preprocessed_files(self):
        return self.preprocessed_dir().iterdir() if self.preprocessed_dir().is_dir() else []

    def preprocessed_file(self, sequence: str):
        return self.preprocessed_dir() / (sequence + ".pth")


class Forces:
    CELL_AREAS = {
        4: [9.461907386779785, 8.415483474731445, 10.215210914611816, 10.031301498413086, 10.03982162475586, 9.951156616210938, 13.936635971069336, 13.902280807495117, 11.974284172058105, 9.510015487670898, 9.57705020904541, 9.693796157836914, 7.969557762145996, 7.802539825439453, 9.645974159240723, 5.009629726409912],
        6: [11.304717063903809, 10.080912590026855, 12.208270072937012, 11.987248420715332, 11.995275497436523, 11.900033950805664, 16.465539932250977, 16.648794174194336, 14.309170722961426, 11.36436939239502, 11.437858581542969, 11.5789794921875, 9.409321784973145, 9.276507377624512, 11.4599609375, 5.939790725708008],
    }
    FRONT_CELLS = [8, 9, 10, 11, 12, 13, 14, 15]
    BACK_CELLS = [0, 1, 2, 3]

    @classmethod
    def from_pressures(cls, pressures, insoles_size, weight):
        areas = torch.as_tensor(cls.CELL_AREAS[insoles_size]).to(pressures)
        return areas * pressures / (9.81 * weight)

    @classmethod
    def gather(cls, forces):														# [...] x F x LR x 16
        cell_groups = [cls.FRONT_CELLS, cls.BACK_CELLS]
        group_forces = [forces[..., cells].sum(
            dim=-1) for cells in cell_groups]  # [...] x F x LR
        return torch.stack(group_forces, dim=-2)									# [...] x F x FB x LR


class Contacts:
    JOINTS = [
        ["left_foot", "right_foot"],
        ["left_ankle", "right_ankle"],
    ]
    JIDXS = torch.as_tensor([[TOPOLOGY.index(joint)
                            for joint in joints] for joints in JOINTS])

    @classmethod
    def from_forces(cls, forces):													# [...] x F x LR x 16
        # Smooth input forces
        forces = util.gma(forces, size=5, std=1.667, dim=-
                          3)						# [...] x F x LR x 16

        # Compute raw contacts
        # [...] x F x  1 x LR
        foot_forces = forces.sum(dim=-1).unsqueeze(-2)
        total_force = foot_forces.sum(
            dim=-1, keepdim=True)							# [...] x F x  1 x  1
        loc_forces = Forces.gather(forces)											# [...] x F x FB x LR
        loc_foot_forces = loc_forces.sum(
            dim=-2, keepdim=True)						# [...] x F x  1 x LR
        # [...] x F x FB x LR
        loc_forces *= (total_force / loc_foot_forces).nan_to_num(0.0)
        contacts = (foot_forces >= 0.10) & (
            loc_forces >= 0.05)						# [...] x F x FB x LR

        # Return smoothed version of contacts
        prev = contacts
        while True:
            contacts = util.sma(prev.float(), size=11, dim=-3) > 0.5
            if (contacts == prev).all():
                return contacts
            prev = contacts


class Dataset(util.DictDataset):
    @classmethod
    def files(cls) -> set:
        return set.union(*[set(subject.preprocessed_files()) for subject in Subject.all()])

    @classmethod
    def parse_files(cls, args) -> set:
        """
                subject
                sequence
                sequence type
                *
        """
        subj_regex = "(?P<subj>(S[1-9][0-9]*)|\*)"
        seq_regex = "(?P<seq>([a-zA-Z0-9]+(-[1-9])?)|\*)"
        item_regex = "(" + subj_regex + "-" + seq_regex + ")"

        files = set()
        for arg in args:
            if arg == "*":
                files |= cls.files()
                continue

            subj_match = re.fullmatch(subj_regex, arg)
            if subj_match is not None:
                subj = subj_match["subj"]
                files |= {src for s in Subject.all()
                          for src in s.preprocessed_files() if subj in [s.id, "*"]}
                continue

            seq_match = re.fullmatch(seq_regex, arg)
            if subj_match is not None:
                seq = seq_match["seq"]
                files |= {src for s in Subject.all()
                          for src in s.preprocessed_files() if seq in [src.stem, "*"]}
                continue

            item_match = re.fullmatch(item_regex, arg)
            if item_match is not None:
                subj, seq = item_match["subj"], item_match["seq"]
                files |= {src for s in Subject.all() for src in s.preprocessed_files() if subj in [
                    s.id, "*"] and seq in [src.stem, "*"]}
                continue

        return files

    @classmethod
    def trainset(cls, *args, train_ids):
        print(train_ids)
        files = cls.parse_files(["*"] if len(args) == 0 else args)
        trainset = cls(*map(str, train_ids))
        return trainset[[index for index, item in enumerate(trainset) if item["file"] in files]]

    @classmethod
    def testset(cls, *args, test_id):
        print(test_id)
        files = cls.parse_files(["*"] if len(args) == 0 else args)
        testset = cls(*map(str, test_id))
        return testset[[index for index, item in enumerate(testset) if item["file"] in files]]

    def __init__(self, *args):
        args = ["*"] if len(args) == 0 else args
        if len(args) == 1 and not isinstance(args[0], str):
            items = args[0]
        else:
            # parse data files to be loaded
            files = self.parse_files(args)

            # load selected data files
            items = []
            for file in sorted(files):
                # Load data, get metadata & map representation
                item = torch.load(file, weights_only=False)
                item["subject"] = file.parent.parent.stem
                item["file"] = file
                items.append(item)
        super().__init__(items)

    def __getitem__(self, arg):
        output = super().__getitem__(arg)
        if isinstance(output, util.DictDataset):
            items = [item for item in output]
            return Dataset(items)
        else:
            return output

    def slices(self, item, starts, stops):
        # parse item
        if isinstance(item, int):
            item = self[item]

        entries, nframes = item.items(), item["poses"].shape[-3]
        # entries, nframes = item.items(), item["positions"].shape[-3]
        # entries, nframes = item.items(), item["angles"].shape[-3]
        items = [dict(entries) for _ in starts]
        for key, value in entries:
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == nframes:
                slices = [value[start:stop]
                          for start, stop in zip(starts, stops)]
            else:
                continue
            for item, window in zip(items, slices):
                item[key] = window
        return items

    def slice(self, item, start=None, stop=None):
        return self.slices(item, [start], [stop])[0]

    def windowed(self, length: int, overlap=0):
        window_step = max(1, length - overlap)
        windows = []
        for index, item in enumerate(self):
            # nframes = item["angles"].shape[-3]
            # nframes = item["positions"].shape[-3]
            nframes = item["poses"].shape[-3]
            starts, stops = torch.arange(nframes).unfold(
                0, length, window_step)[:, [0, -1]].unbind(-1)
            windows += self.slices(index, starts, stops+1)
        return self.__class__(windows)
