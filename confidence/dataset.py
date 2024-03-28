import math
import os
import traceback
import pickle
import random
from argparse import Namespace
from functools import partial
import copy

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.pdbbind import PDBBind, model_conf_to_pdb_args
from utils.diffusion_utils import get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def get_cache_path(args, split):
    cache_path = args.cache_path
    if not args.no_torsion:
        cache_path += '_torsion'
    if args.all_atoms:
        cache_path += '_allatoms'
    split_path = args.split_train if split == 'train' else args.split_val
    cache_path = os.path.join(cache_path, f'limit{args.limit_complexes}_INDEX{os.path.splitext(os.path.basename(split_path))[0]}_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}'
                                       + ('' if not args.all_atoms else f'_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}')
                                       + ('' if args.no_torsion or args.num_conformers == 1 else
                                           f'_confs{args.num_conformers}')
                              + ('' if args.esm_embeddings_path is None else f'_esmEmbeddings'))
    return cache_path

def get_args_and_cache_path(original_model_dir, split):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
    return model_args, get_cache_path(model_args,split)



class ConfidenceDataset(Dataset):
    PREPROCESSING_ERRORS_FP = "conf_dataset_errors.txt"
    def __init__(self, cache_path, original_model_dir, split, device, limit_complexes,
                 inference_steps, samples_per_complex, all_atoms,
                 args, model_ckpt, balance=False, use_original_model_cache=True, rmsd_classification_cutoff=2,
                 cache_ids_to_combine=None, cache_creation_id=None,
                 deterministic_sample=False
            ):

        super(ConfidenceDataset, self).__init__()

        self.device = device
        self.inference_steps = inference_steps
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.model_ckpt = model_ckpt
        self.deterministic_sample = deterministic_sample
        assert not (
            self.deterministic_sample and self.balance
        ), "These parameters cannot be combined"

        self.original_model_args, _ = get_args_and_cache_path(original_model_dir, split)

        # check if the docked positions have already been computed, if not run the preprocessing (docking every complex)
        self.full_cache_path = os.path.join(cache_path, f'model_{os.path.splitext(os.path.basename(original_model_dir))[0]}'
                                            f'_split_{split}_limit_{limit_complexes}')

        print(f'Loading or recreating original model heterographs')
        original_model_pdb_args = model_conf_to_pdb_args(
            self.original_model_args,
            split_path=self.original_model_args.split_val if split == 'val' else self.original_model_args.split_train,
            keep_original=True,
            require_ligand=True,
            transform=None,
            num_workers=1,
        )
        original_model_pdbbind_dataset = PDBBind(
            **original_model_pdb_args,
        )

        if (
            (not os.path.exists(os.path.join(self.full_cache_path, "ligand_positions.pkl"))
             and self.cache_creation_id is None)
             or (not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{self.cache_creation_id}.pkl"))
                 and self.cache_creation_id is not None
                 )
        ):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing(original_model_pdbbind_dataset)

        # load the graphs that the confidence model will use
        if self.use_original_model_cache:
            print('Using the cached complex graphs of the original model args')
            self.complex_graphs_cache = original_model_pdbbind_dataset.full_cache_path
            print(self.complex_graphs_cache)
        else:
            print('Not using the cached complex graphs of the original model args. Instead the complex graphs are used that are at the location given by the dataset parameters given to confidence_train.py')
            conf_model_pdb_args = model_conf_to_pdb_args(
                args,
                split_path=args.split_val if split == 'val' else args.split_train,
                keep_original=True,
                require_ligand=True,
                transform=None,
                num_workers=1,
            )
            pdbbind_dataset = PDBBind(
                **conf_model_pdb_args,
            )
            self.pdbbind_dataset = pdbbind_dataset

        if self.cache_ids_to_combine is None:
            print(f'HAPPENING | Loading positions and rmsds from: {os.path.join(self.full_cache_path, "ligand_positions.pkl")}')
            with open(os.path.join(self.full_cache_path, "ligand_positions.pkl"), 'rb') as f:
                self.full_ligand_positions, self.rmsds = pickle.load(f)
            if os.path.exists(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl")):
                with open(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"), 'rb') as f:
                    generated_rmsd_complex_names = pickle.load(f)
            else:
                print('HAPPENING | The path, ', os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"),
                      ' does not exist. \n => We assume that means that we are using a ligand_positions.pkl where the '
                      'code was not saving the complex names for them yet. We now instead use the complex names of '
                      'the dataset that the original model used to create the ligand positions and RMSDs.')
                num_complexes = original_model_pdbbind_dataset.len()
                generated_rmsd_complex_names = [
                    original_model_pdbbind_dataset.get(i) for i in range(num_complexes)
                    ]
            assert (len(self.rmsds) == len(generated_rmsd_complex_names))
        else:
            all_rmsds_unsorted, all_full_ligand_positions_unsorted, all_names_unsorted = [], [], []
            for idx, cache_id in enumerate(self.cache_ids_to_combine):
                print(f'HAPPENING | Loading positions and rmsds from cache_id from the path: {os.path.join(self.full_cache_path, "ligand_positions_"+ str(cache_id)+ ".pkl")}')
                if not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl")): raise Exception(f'The generated ligand positions with cache_id do not exist: {cache_id}') # be careful with changing this error message since it is sometimes cought in a try catch
                with open(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl"), 'rb') as f:
                    full_ligand_positions, rmsds = pickle.load(f)
                with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"), 'rb') as f:
                    names_unsorted = pickle.load(f)
                all_names_unsorted.append(names_unsorted)
                all_rmsds_unsorted.append(rmsds)
                all_full_ligand_positions_unsorted.append(full_ligand_positions)
            names_order = list(set(sum(all_names_unsorted, [])))
            all_rmsds, all_full_ligand_positions, all_names = [], [], []
            for idx, (rmsds_unsorted, full_ligand_positions_unsorted, names_unsorted) in enumerate(zip(all_rmsds_unsorted,all_full_ligand_positions_unsorted, all_names_unsorted)):
                name_to_pos_dict = {name: (rmsd, pos) for name, rmsd, pos in zip(names_unsorted, full_ligand_positions_unsorted, rmsds_unsorted) }
                intermediate_rmsds = [name_to_pos_dict[name][1] for name in names_order]
                all_rmsds.append((intermediate_rmsds))
                intermediate_pos = [name_to_pos_dict[name][0] for name in names_order]
                all_full_ligand_positions.append((intermediate_pos))
            self.full_ligand_positions, self.rmsds = [], []
            for positions_tuple in list(zip(*all_full_ligand_positions)):
                self.full_ligand_positions.append(np.concatenate(positions_tuple, axis=0))
            for positions_tuple in list(zip(*all_rmsds)):
                self.rmsds.append(np.concatenate(positions_tuple, axis=0))
            generated_rmsd_complex_names = names_order
        print('Number of complex graphs: ', self.pdbbind_dataset.len())
        print('Number of RMSDs and positions for the complex graphs: ', len(self.full_ligand_positions))

        self.all_samples_per_complex = samples_per_complex * (1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine))

        print(f'HAPPENING | Creating complex name to dataset id map')
        self.complex_name_to_id_dict = {}
        for id_ in range(pdbbind_dataset.len()):
            complex = pdbbind_dataset.get(id_)
            self.complex_name_to_id_dict[complex.name] = id_

        self.positions_rmsds_dict = {name: (pos, rmsd) for name, pos, rmsd in zip (generated_rmsd_complex_names, self.full_ligand_positions, self.rmsds)}
        self.dataset_names = list(set(self.positions_rmsds_dict.keys()) & set(self.complex_name_to_id_dict.keys()))
        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]
        if self.deterministic_sample:
            self.dataset_names = sorted(self.dataset_names)

    def len(self):
        return len(self.dataset_names)

    def get(self, idx):
        complex_name = self.dataset_names[idx]
        pdbbind_dataset_id = self.complex_name_to_id_dict[complex_name]
        complex_graph = copy.deepcopy(self.pdbbind_dataset.get(pdbbind_dataset_id))
        positions, rmsds = self.positions_rmsds_dict[complex_name]

        # Filtering out nan rmsds
        keep_idx = np.isfinite(rmsds)
        positions = positions[keep_idx, :]
        rmsds = rmsds[keep_idx]
        num_samples = sum(keep_idx)

        if self.balance:
            if isinstance(self.rmsd_classification_cutoff, list): raise ValueError("a list for --rmsd_classification_cutoff can only be used without --balance")
            label = random.randint(0, 1)
            success = rmsds < self.rmsd_classification_cutoff
            n_success = np.count_nonzero(success)
            if label == 0 and n_success != num_samples:
                # sample negative complex
                sample = random.randint(0, num_samples - n_success - 1)
                lig_pos = positions[~success][sample]
                complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
            else:
                # sample positive complex
                if n_success > 0: # if no successfull sample returns the matched complex
                    sample = random.randint(0, n_success - 1)
                    lig_pos = positions[success][sample]
                    complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
            complex_graph.y = torch.tensor(label).float()
        else:
            if not self.deterministic_sample:
                sample = random.randint(0, num_samples - 1)
            else:
                sample = idx % num_samples
            complex_graph['ligand'].pos = torch.from_numpy(positions[sample])
            complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff).float().unsqueeze(0)
            if isinstance(self.rmsd_classification_cutoff, list):
                complex_graph.y_binned = torch.tensor(np.logical_and(rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],rmsds[sample] >= [0] + self.rmsd_classification_cutoff), dtype=torch.float).unsqueeze(0)
                complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0]).unsqueeze(0).float()
            complex_graph.rmsd = torch.tensor(rmsds[sample]).unsqueeze(0).float()

        complex_graph['ligand'].node_t = {'tr': 0 * torch.ones(complex_graph['ligand'].num_nodes),
                                          'rot': 0 * torch.ones(complex_graph['ligand'].num_nodes),
                                          'tor': 0 * torch.ones(complex_graph['ligand'].num_nodes)}
        complex_graph['receptor'].node_t = {'tr': 0 * torch.ones(complex_graph['receptor'].num_nodes),
                                            'rot': 0 * torch.ones(complex_graph['receptor'].num_nodes),
                                            'tor': 0 * torch.ones(complex_graph['receptor'].num_nodes)}
        if self.all_atoms:
            complex_graph['atom'].node_t = {'tr': 0 * torch.ones(complex_graph['atom'].num_nodes),
                                            'rot': 0 * torch.ones(complex_graph['atom'].num_nodes),
                                            'tor': 0 * torch.ones(complex_graph['atom'].num_nodes)}
        complex_graph.complex_t = {'tr': 0 * torch.ones(1), 'rot': 0 * torch.ones(1), 'tor': 0 * torch.ones(1)}
        return complex_graph

    def preprocessing(
            self,
            original_model_pdbbind_dataset: PDBBind,
        ):
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)

        model = get_model(self.original_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(f'{self.original_model_dir}/{self.model_ckpt}', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()

        tr_schedule = get_t_schedule(
            sigma_schedule=self.original_model_args.sigma_schedule,
            inference_steps=self.inference_steps
        )
        rot_schedule = tr_schedule
        tor_schedule = tr_schedule
        print('common t schedule', tr_schedule)

        print('HAPPENING | Creating DataLoader for old dataset')
        # dataset = ListDataset(complex_graphs)
        loader = DataLoader(
            dataset=original_model_pdbbind_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )

        cache_id_str = '' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)
        complex_results_cache_folder = os.path.join(
            self.full_cache_path,
            f"complex_cache{cache_id_str}/"
        )
        os.makedirs(complex_results_cache_folder, exist_ok=True)
        rmsds, full_ligand_positions, names = [], [], []
        start_batch_size = self.samples_per_complex
        for orig_complex_graph in tqdm(
            loader,
            total=original_model_pdbbind_dataset.len()
        ):
            assert(len(orig_complex_graph.name) == 1), "there should only be one name"
            name = orig_complex_graph.name[0]
            complex_pickle_fp = os.path.join(
                complex_results_cache_folder,
                f"{name}.pkl"
            )
            if os.path.exists(complex_pickle_fp):
                with open(complex_pickle_fp, "rb") as f:
                    (complex_full_ligand_positions, rmsd) = pickle.load(f)
                rmsds.append(rmsd)
                full_ligand_positions.append(complex_full_ligand_positions)
                names.append(name)
                continue

            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
            randomize_position(data_list, self.original_model_args.no_torsion, False, self.original_model_args.tr_sigma_max)

            predictions_list = None
            batch_size = start_batch_size
            failed_convergence_counter = 0
            other_exception_counter = 0
            use_start_ligand_position_as_data = False
            while predictions_list is None:
                try:
                    predictions_list, confidences = sampling(
                        data_list=data_list,
                        model=model,
                        inference_steps=self.inference_steps,
                        tr_schedule=tr_schedule,
                        rot_schedule=rot_schedule,
                        tor_schedule=tor_schedule,
                        device=self.device,
                        t_to_sigma=t_to_sigma,
                        model_args=self.original_model_args,
                        batch_size=batch_size,
                    )
                except Exception as e:
                    if 'failed to converge' in str(e):
                        failed_convergence_counter += 1
                        if failed_convergence_counter > 5:
                            print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                            use_start_ligand_position_as_data = True
                            break
                        print('| WARNING: SVD failed to converge - trying again with a new sample')
                    elif 'CUDA out of memory.' in str(e):
                        batch_size = batch_size // 2
                        print(f'| WARNING: CUDA OOM - trying with batch_size of {batch_size}')
                        if batch_size < 2:
                            print('| WARNING: CUDA OOM if bs >1 - skipping the complex')
                            use_start_ligand_position_as_data = True
                            break
                    else:
                        other_exception_counter += 1
                        if other_exception_counter > 5:
                            print(f"| WARNING: exceptions 5 times - skipping the complex")
                            use_start_ligand_position_as_data = True
                            error_log_fp = os.path.join(
                                self.full_cache_path,
                                self.PREPROCESSING_ERRORS_FP
                            )
                            with open(error_log_fp, "at") as f:
                                f.write(f"\n\nError when processing {name}\n")
                                f.write(f"{str(e)}\n")
                                f.write(f"{traceback.format_exc()}\n")
                            break
                        print(f"| WARNING: encountered exception {e} trying again.")
            if use_start_ligand_position_as_data:
                predictions_list = data_list
            if self.original_model_args.no_torsion:
                orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

            filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

            if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

            ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
            orig_ligand_pos = np.expand_dims(orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

            rmsds.append(rmsd)
            complex_full_ligand_positions = np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in predictions_list])
            full_ligand_positions.append(complex_full_ligand_positions)
            names.append(name)
            with open(complex_pickle_fp, "wb") as f:
                pickle.dump((complex_full_ligand_positions, rmsd), f)

        ligand_pos_fp = os.path.join(self.full_cache_path, f"ligand_positions{cache_id_str}.pkl")
        with open(ligand_pos_fp, 'wb') as f:
            pickle.dump((full_ligand_positions, rmsds), f)
        complex_names_fp = os.path.join(self.full_cache_path, f"complex_names_in_same_order{cache_id_str}.pkl")
        with open(complex_names_fp, 'wb') as f:
            pickle.dump((names), f)