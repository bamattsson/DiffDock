import binascii
import glob
import os
import pickle
from typing import Dict, Any
from collections import defaultdict
import random
import copy
import torch.nn.functional as F
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from rdkit.Chem import RemoveAllHs
from types import SimpleNamespace

from datasets.process_mols import read_molecule, get_lig_graph_with_matching, generate_conformer, moad_extract_receptor_structure
from utils.diffusion_utils import modify_conformer, set_time
from utils.utils import read_strings_from_txt, crop_beyond
from utils import so3, torus


def model_conf_to_pdb_args(
        model_conf_args,
        split_path,
        keep_original=True,
        require_ligand=False,
        transform=None,
        num_workers=None,
    ) -> Dict:
    """Get arguments to instantiate PDB."""

    args_from_model_conf = {
        "limit_complexes": model_conf_args.limit_complexes,
        'chain_cutoff': model_conf_args.chain_cutoff,
        "receptor_radius": model_conf_args.receptor_radius,
        "c_alpha_max_neighbors": model_conf_args.c_alpha_max_neighbors,
        "remove_hs": model_conf_args.remove_hs,
        "max_lig_size": model_conf_args.max_lig_size,
        "matching": not model_conf_args.no_torsion,
        "popsize": model_conf_args.matching_popsize,
        "maxiter": model_conf_args.matching_maxiter,
        "all_atoms": model_conf_args.all_atoms,
        "atom_radius": model_conf_args.atom_radius,
        "atom_max_neighbors": model_conf_args.atom_max_neighbors,
        "knn_only_graph": not getattr(model_conf_args, 'not_knn_only_graph', True),
        "include_miscellaneous_atoms": getattr(model_conf_args, 'include_miscellaneous_atoms', False),
        "matching_tries": model_conf_args.matching_tries,
        "num_conformers": model_conf_args.num_conformers,
        "root": model_conf_args.data_dir,
        "cache_path": model_conf_args.cache_path,
        "esm_embeddings_path": model_conf_args.esm_embeddings_path,
        "num_workers": int(getattr(model_conf_args, "num_workers", 1))
    }

    all_args = {
        **args_from_model_conf,
        "split_path": split_path,
        "keep_original": keep_original,
        "transform": transform,
        "require_ligand": require_ligand,
    }

    if num_workers is not None:
        all_args["num_workers"] = num_workers

    return all_args


class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom, alpha=1, beta=1,
                 include_miscellaneous_atoms=False, crop_beyond_cutoff=None, time_independent=False, rmsd_cutoff=0,
                 minimum_t=0, sampling_mixing_coeff=0):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.minimum_t = minimum_t
        self.mixing_coeff = sampling_mixing_coeff
        self.alpha = alpha
        self.beta = beta
        self.crop_beyond_cutoff = crop_beyond_cutoff
        self.rmsd_cutoff = rmsd_cutoff
        self.time_independent = time_independent

    def __call__(self, data):
        t_tr, t_rot, t_tor, t = self.get_time()
        return self.apply_noise(data, t_tr, t_rot, t_tor, t)

    def get_time(self):
        if self.time_independent:
            t = np.random.beta(self.alpha, self.beta)
            t_tr, t_rot, t_tor = t,t,t
        else:
            t = None
            if self.mixing_coeff == 0:
                t = np.random.beta(self.alpha, self.beta)
                t = self.minimum_t + t * (1 - self.minimum_t)
            else:
                choice = np.random.binomial(1, self.mixing_coeff)
                t1 = np.random.beta(self.alpha, self.beta)
                t1 = t1 * self.minimum_t
                t2 = np.random.beta(self.alpha, self.beta)
                t2 = self.minimum_t + t2 * (1 - self.minimum_t)
                t = choice * t1 + (1 - choice) * t2 

            t_tr, t_rot, t_tor = t,t,t
        return t_tr, t_rot, t_tor, t

    def apply_noise(self, data, t_tr, t_rot, t_tor, t, tr_update = None, rot_update=None, torsion_updates=None):
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)

        if self.time_independent:
            orig_complex_graph = copy.deepcopy(data)

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)

        if self.time_independent:
            set_time(data, 0, 0, 0, 0, 1, self.all_atom, device=None, include_miscellaneous_atoms=self.include_miscellaneous_atoms)
        else:
            set_time(data, t, t_tr, t_rot, t_tor, 1, self.all_atom, device=None, include_miscellaneous_atoms=self.include_miscellaneous_atoms)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
        torsion_updates = None if self.no_torsion else torsion_updates
        try:
            modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)
        except Exception as e:
            print("failed modify conformer")
            print(e)

        if self.time_independent:
            if self.no_torsion:
                orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

            filterHs = torch.not_equal(data['ligand'].x[:, 0], 0).cpu().numpy()
            if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
            ligand_pos = data['ligand'].pos.cpu().numpy()[filterHs]
            orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy()
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=1).mean(axis=0))
            data.y = torch.tensor(rmsd < self.rmsd_cutoff).float().unsqueeze(0)
            data.atom_y = data.y
            return data

        data.tr_score = -tr_update / tr_sigma ** 2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma

        if data['ligand'].pos.shape[0] == 1:
            # if the ligand is a single atom, the rotational score is always 0
            data.rot_score = data.rot_score * 0

        if self.crop_beyond_cutoff is not None:
            crop_beyond(data, tr_sigma * 3 + self.crop_beyond_cutoff, self.all_atom)
        set_time(data, t, t_tr, t_rot, t_tor, 1, self.all_atom, device=None, include_miscellaneous_atoms=self.include_miscellaneous_atoms)
        return data


HETEROGRAPH_CACHE_SUBFOLDER = "heterographs_folder"
RDKIT_LIGAND_CACHE_SUBFOLDER = "rdkit_ligands_folder"


class PDBBind(Dataset):

    HETEROGRAPH_LIST_FP = "heterograph_fps.txt"
    RDKIT_LIGAND_LIST_FP = "rdkit_ligand_fps.txt"
    def __init__(
            self,
            root,
            transform=None,
            cache_path='data/cache',
            split_path='data/',
            limit_complexes=0,
            chain_cutoff=10,
            receptor_radius=30,
            num_workers=1,
            c_alpha_max_neighbors=None,
            popsize=15,
            maxiter=15,
            matching=True,
            keep_original=False,
            max_lig_size=None,
            remove_hs=False,
            num_conformers=1,
            all_atoms=False,
            atom_radius=5,
            atom_max_neighbors=None,
            esm_embeddings_path=None,
            require_ligand=False,
            include_miscellaneous_atoms=False,
            protein_path_list=None,
            ligand_descriptions=None,
            keep_local_structures=False,
            protein_file="protein_processed",
            ligand_file="ligand",
            knn_only_graph=False,
            matching_tries=1,
            dataset='PDBBind'
        ):

        super(PDBBind, self).__init__(root, transform)
        self.pdbbind_dir = root
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.chain_cutoff = chain_cutoff
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.use_old_wrong_embedding_order = False
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.protein_file = protein_file
        self.fixed_knn_radius_graph = True
        self.knn_only_graph = knn_only_graph
        self.matching_tries = matching_tries
        self.ligand_file = ligand_file
        self.dataset = dataset
        assert knn_only_graph or (not all_atoms)
        self.all_atoms = all_atoms
        if matching or protein_path_list is not None and ligand_descriptions is not None:
            cache_path += '_torsion'
        if all_atoms:
            cache_path += '_allatoms'
        self.full_cache_path = os.path.join(cache_path, f'{dataset}3_limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                                        f'_chainCutoff{self.chain_cutoff if self.chain_cutoff is None else int(self.chain_cutoff)}'
                                            + (''if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + (''if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            + ('' if self.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + '_full'
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode())))
                                            + ('' if protein_file == "protein_processed" else '_' + protein_file)
                                            + ('' if not self.fixed_knn_radius_graph else (f'_fixedKNN' if not self.knn_only_graph else '_fixedKNNonly'))
                                            + ('' if not self.include_miscellaneous_atoms else '_miscAtoms')
                                            + ('' if self.use_old_wrong_embedding_order else '_chainOrd')
                                            + ('' if self.matching_tries == 1 else f'_tries{matching_tries}'))
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers

        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        if not self.check_all_complexes():
            os.makedirs(self.full_cache_path, exist_ok=True)
            os.makedirs(
                os.path.join(self.full_cache_path, HETEROGRAPH_CACHE_SUBFOLDER),
                exist_ok=True
            )
            os.makedirs(
                os.path.join(self.full_cache_path, RDKIT_LIGAND_CACHE_SUBFOLDER),
                exist_ok=True
            )
            if protein_path_list is None or ligand_descriptions is None:
                self.preprocessing()
            else:
                self.inference_preprocessing()
        else:
            with open(os.path.join(
                self.full_cache_path,
                self.HETEROGRAPH_LIST_FP,
                ), 'rt'
            ) as f:
                self.complex_graph_fps = f.read().split("\n")

            with open(os.path.join(
                self.full_cache_path,
                self.RDKIT_LIGAND_LIST_FP,
                ), 'rt'
            ) as f:
                self.rdkit_ligand_fps = f.read().split("\n")

        print_statistics(self.complex_graph_fps)
        list_names = []
        for complex_graph_fp in self.complex_graph_fps:
            with open(complex_graph_fp, "rb") as f:
                list_names.append(pickle.load(f)[0]["name"])
        list_name_txt_fp = os.path.join(self.full_cache_path, f'pdbbind_{os.path.splitext(os.path.basename(self.split_path))[0][:3]}_names.txt')
        if not os.path.exists(list_name_txt_fp):
            with open(list_name_txt_fp, 'w') as f:
                f.write('\n'.join(list_names))

    def len(self):
        return len(self.complex_graph_fps)

    def get(self, idx):
        with open(self.complex_graph_fps[idx], "rb") as f:
            complex_graph = pickle.load(f)[0]
        if self.require_ligand:
            with open(self.rdkit_ligand_fps[idx], "rb") as f:
                mol = pickle.load(f)[0]
            complex_graph.mol = RemoveAllHs(mol)

        for a in ['random_coords', 'coords', 'seq', 'sequence', 'mask', 'rmsd_matching', 'cluster', 'orig_seq', 'to_keep', 'chain_ids']:
            if hasattr(complex_graph, a):
                delattr(complex_graph, a)
            if hasattr(complex_graph['receptor'], a):
                delattr(complex_graph['receptor'], a)

        return complex_graph

    def preprocessing(self):
        print(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        print(f'Loading {len(complex_names_all)} complexes.')

        assert len(complex_names_all) == len(set(complex_names_all)), "names need to be unique"

        if self.esm_embeddings_path is not None:
            id_to_embeddings = torch.load(self.esm_embeddings_path)
            chain_embeddings_dictlist = defaultdict(list)
            chain_indices_dictlist = defaultdict(list)
            for key, embedding in id_to_embeddings.items():
                key_name = key.split('_chain_')[0]
                if key_name in complex_names_all:
                    chain_embeddings_dictlist[key_name].append(embedding)
                    chain_indices_dictlist[key_name].append(int(key.split('_chain_')[1]))
            lm_embeddings_chains_all = []
            for name in complex_names_all:
                complex_chains_embeddings = chain_embeddings_dictlist[name]
                complex_chains_indices = chain_indices_dictlist[name]
                chain_reorder_idx = np.argsort(complex_chains_indices)
                reordered_chains = [complex_chains_embeddings[i] for i in chain_reorder_idx]
                lm_embeddings_chains_all.append(reordered_chains)
        else:
            print("WARN | esm_embeddings were not available, adding None instead")
            lm_embeddings_chains_all = [None] * len(complex_names_all)

        # Processing all the claims, dumping pickled outputs in a temporary folder
        self.preprocess_all_remaining_complexes(
            complex_names_all,
            complex_names_all,
            [None] * len(complex_names_all),
            [None] * len(complex_names_all),
            lm_embeddings_chains_all,
        )

    def inference_preprocessing(self):
        ligands_list = []
        print('Reading molecules and generating local structures with RDKit')
        for ligand_description in tqdm(self.ligand_descriptions):
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path
            if mol is not None:
                mol = AddHs(mol)
                generate_conformer(mol)
                ligands_list.append(mol)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if not self.keep_local_structures:
                    mol.RemoveAllConformers()
                    mol = AddHs(mol)
                    generate_conformer(mol)
                ligands_list.append(mol)

        if self.esm_embeddings_path is not None:
            print('Reading language model embeddings.')
            lm_embeddings_chains_all = []
            if not os.path.exists(self.esm_embeddings_path): raise Exception('ESM embeddings path does not exist: ',self.esm_embeddings_path)
            for protein_path in self.protein_path_list:
                embeddings_paths = sorted(glob.glob(os.path.join(self.esm_embeddings_path, os.path.basename(protein_path)) + '*'))
                lm_embeddings_chains = []
                for embeddings_path in embeddings_paths:
                    lm_embeddings_chains.append(torch.load(embeddings_path)['representations'][33])
                lm_embeddings_chains_all.append(lm_embeddings_chains)
        else:
            lm_embeddings_chains_all = [None] * len(self.protein_path_list)

        sample_names = [str(num) for num in range(self.ligand_descriptions)]

        self.preprocess_all_remaining_complexes(
            sample_names,
            self.protein_path_list,
            self.ligand_descriptions,
            ligands_list,
            lm_embeddings_chains_all,
        )

    def preprocess_all_remaining_complexes(
            self,
            cache_names,
            protein_paths,
            ligand_descriptions,
            ligands_list,
            lm_embeddings_chains_all,
    ):
        """
        Processing all the claims, saving in a temporary folder, aggregating all into final outputs.
        """
        print('Generating graphs for ligands and proteins')

        sample_names_remaining = []
        protein_paths_remaining = []
        ligand_descriptions_remaining = []
        ligands_remaining = []
        lm_embeddings_chains_remaining = []

        complex_graph_fps, rdkit_ligand_fps = [], []
        for cache_name, protein_path, ligand_desc, ligand, lm_embeddings_chains in zip(
            cache_names,
            protein_paths,
            ligand_descriptions,
            ligands_list,
            lm_embeddings_chains_all,
        ):
            heterograph_fp = os.path.join(
                self.full_cache_path,
                HETEROGRAPH_CACHE_SUBFOLDER,
                f"{cache_name}.pkl"
            )
            rdkit_ligand_fp = os.path.join(
                self.full_cache_path,
                RDKIT_LIGAND_CACHE_SUBFOLDER,
                f"{cache_name}.pkl"
            )
            if os.path.exists(heterograph_fp) and os.path.exists(rdkit_ligand_fp):
                complex_graph_fps.append(heterograph_fp)
                rdkit_ligand_fps.append(rdkit_ligand_fp)
            else:
                sample_names_remaining.append(cache_name)
                protein_paths_remaining.append(protein_path)
                ligand_descriptions_remaining.append(ligand_desc)
                ligands_remaining.append(ligand)
                lm_embeddings_chains_remaining.append(lm_embeddings_chains)

        print(f"Processing remaining {len(sample_names_remaining)} / {len(cache_names)} samples.")

        if self.num_workers > 1:
            # Note: multiprocessing leaks open file_handles, causing the program to be killed
            # unless you set ulimit -n to a large number, e.g. 30000
            # TODO: look into this, possible cause: the passing of lm_embeddings_chains_remaining
            # through the pipe into imap (as the ever increasing amount of open files
            # is from torch)
            ctx = torch.multiprocessing.get_context('spawn')
            p = ctx.Pool(self.num_workers)
            p.__enter__()
            original_torch_mp_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
            if "file_system" in torch.multiprocessing.get_all_sharing_strategies():
                print("setting sharing strategy to file_descriptor")
                torch.multiprocessing.set_sharing_strategy("file_system")
        subset_of_self = get_namespace_with_needed_attributes(self)
        with tqdm(
            total=len(sample_names_remaining),
            desc=f'Loading samples:') as pbar:
            map_fn = p.imap_unordered if self.num_workers > 1 else map
            for t in map_fn(
                process_and_save_complexes,
                zip(
                    [subset_of_self] * len(sample_names_remaining),
                    cache_names,
                    protein_paths_remaining,
                    lm_embeddings_chains_remaining,
                    ligands_remaining,
                    ligand_descriptions_remaining,
                ),
                ):
                complex_graph_fps.extend(t[0])
                rdkit_ligand_fps.extend(t[1])
                pbar.update()
        if self.num_workers > 1:
            p.__exit__(None, None, None)
            torch.multiprocessing.set_sharing_strategy(original_torch_mp_sharing_strategy)

        print(f"Finished processing all complexes.")
        self.complex_graph_fps = complex_graph_fps
        with open(os.path.join(
            self.full_cache_path,
            self.HETEROGRAPH_LIST_FP,
        ), 'wt') as f:
            f.write("\n".join(self.complex_graph_fps))

        self.rdkit_ligand_fps = rdkit_ligand_fps
        with open(os.path.join(
            self.full_cache_path,
            self.RDKIT_LIGAND_LIST_FP,
        ), 'wt') as f:
            f.write("\n".join(self.rdkit_ligand_fps))
        print(f"Saved lists with all processed data points.")

    def check_all_complexes(self):
        if os.path.exists(os.path.join(self.full_cache_path, self.HETEROGRAPH_LIST_FP)):
            return True

    def collect_all_complexes(self):
        print('Collecting all complexes from cache', self.full_cache_path)
        if os.path.exists(os.path.join(self.full_cache_path, f"heterographs.pkl")):
            with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'rb') as f:
                complex_graphs = pickle.load(f)
            if self.require_ligand:
                with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
                    rdkit_ligands = pickle.load(f)
            else:
                rdkit_ligands = None
            return complex_graphs, rdkit_ligands


def get_namespace_with_needed_attributes(pdbbind_obj):

    NEEDED_ATTRIBUTES = [
        "pdbbind_dir",
        "ligand_file",
        "max_lig_size",
        "popsize",
        "maxiter",
        "matching",
        "keep_original",
        "num_conformers",
        "remove_hs",
        "matching_tries",
        "protein_file",
        "receptor_radius",
        "c_alpha_max_neighbors",
        "knn_only_graph",
        "all_atoms",
        "atom_radius",
        "atom_max_neighbors",
        "dataset",
        "full_cache_path",
    ]

    dict_ = {na: getattr(pdbbind_obj, na) for na in NEEDED_ATTRIBUTES}

    return SimpleNamespace(**dict_)


def process_and_save_complexes(
        par,
    ):
    self, cache_name, name, lm_embedding_chains, ligand, ligand_description = par
    heterograph_fp = os.path.join(
        self.full_cache_path,
        HETEROGRAPH_CACHE_SUBFOLDER,
        f"{cache_name}.pkl"
    )
    rdkit_ligand_fp = os.path.join(
        self.full_cache_path,
        RDKIT_LIGAND_CACHE_SUBFOLDER,
        f"{cache_name}.pkl"
    )
    if os.path.exists(heterograph_fp) and os.path.exists(rdkit_ligand_fp):
        return [heterograph_fp], [rdkit_ligand_fp]
    if not os.path.exists(os.path.join(self.pdbbind_dir, name)) and ligand is None:
        print("Folder not found", name)
        return [], []

    try:

        lig = read_mol(self.pdbbind_dir, name, suffix=self.ligand_file, remove_hs=False)
        if self.max_lig_size != None and lig.GetNumHeavyAtoms() > self.max_lig_size:
            print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
            return [], []

        complex_graph = HeteroData()
        complex_graph['name'] = name
        get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                    self.num_conformers, remove_hs=self.remove_hs, tries=self.matching_tries)

        moad_extract_receptor_structure(path=os.path.join(self.pdbbind_dir, name, f'{name}_{self.protein_file}.pdb'),
                                        complex_graph=complex_graph,
                                        neighbor_cutoff=self.receptor_radius,
                                        max_neighbors=self.c_alpha_max_neighbors,
                                        lm_embeddings=lm_embedding_chains,
                                        knn_only_graph=self.knn_only_graph,
                                        all_atoms=self.all_atoms,
                                        atom_cutoff=self.atom_radius,
                                        atom_max_neighbors=self.atom_max_neighbors)

    except Exception as e:
        print(f'Skipping {name} because of the error:')
        print(e)
        return [], []

    if self.dataset == 'posebusters':
        other_positions = []
        all_mol_file = os.path.join(self.pdbbind_dir, name, f'{name}_ligands.sdf')
        supplier = Chem.SDMolSupplier(all_mol_file, sanitize=False, removeHs=False)
        for mol in supplier:
            Chem.SanitizeMol(mol)
            all_mol = RemoveAllHs(mol)
            for conf in all_mol.GetConformers():
                other_positions.append(conf.GetPositions())

        print(f'Found {len(other_positions)} alternative poses for {name}')
        complex_graph['ligand'].orig_pos = np.asarray(other_positions)

    protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
    complex_graph['receptor'].pos -= protein_center
    if self.all_atoms:
        complex_graph['atom'].pos -= protein_center

    if (not self.matching) or self.num_conformers == 1:
        complex_graph['ligand'].pos -= protein_center
    else:
        for p in complex_graph['ligand'].pos:
            p -= protein_center

    complex_graph.original_center = protein_center
    complex_graph['receptor_name'] = name

    with open(heterograph_fp, "wb") as f:
        pickle.dump([complex_graph], f)
    with open(rdkit_ligand_fp, "wb") as f:
        pickle.dump([lig], f)

    return [heterograph_fp], [rdkit_ligand_fp]


def print_statistics(complex_graph_fps):
    statistics = ([], [], [], [], [], [])
    receptor_sizes = []

    for complex_graph_fp in tqdm(complex_graph_fps, "Calculating statistics"):
        with open(complex_graph_fp, 'rb') as f:
            complex_graph = pickle.load(f)[0]
        lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
        receptor_sizes.append(complex_graph['receptor'].pos.shape[0])
        radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
        molecule_center = torch.mean(lig_pos, dim=0)
        radius_molecule = torch.max(
            torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(molecule_center)
        statistics[0].append(radius_protein)
        statistics[1].append(radius_molecule)
        statistics[2].append(distance_center)
        if "rmsd_matching" in complex_graph:
            statistics[3].append(complex_graph.rmsd_matching)
        else:
            statistics[3].append(0)
        statistics[4].append(int(complex_graph.random_coords) if "random_coords" in complex_graph else -1)
        if "random_coords" in complex_graph and complex_graph.random_coords and "rmsd_matching" in complex_graph:
            statistics[5].append(complex_graph.rmsd_matching)

    if len(statistics[5]) == 0:
        statistics[5].append(-1)
    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching', 'random coordinates', 'random rmsd matching']
    print('Number of complexes: ', len(complex_graph_fps))
    for i in range(len(name)):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")

    return


def read_mol(pdbbind_dir, name, suffix='ligand', remove_hs=False):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_{suffix}.sdf'), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_{suffix}.mol2'), remove_hs=remove_hs, sanitize=True)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs