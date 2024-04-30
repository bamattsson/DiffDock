import copy
import os
import torch
import traceback

from datasets.moad import MOAD
from utils.gnina_utils import get_gnina_poses
from utils.molecules_utils import get_symmetry_rmsd

torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import time
from argparse import ArgumentParser, Namespace, FileType
from datetime import datetime
from functools import partial
import numpy as np
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from rdkit.Chem import RemoveAllHs

from datasets.pdbbind import PDBBind, model_conf_to_pdb_args
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model, ExponentialMovingAverage, save_yaml_file
from utils.visualise import PDBFile
from tqdm import tqdm
from dltools.datasets.heterograph_dataset import HeterographDataset

RDLogger.DisableLog('rdApp.*')
import yaml
import pickle


def get_dataset(args, model_args, heterograph_dataset_args):
    if heterograph_dataset_args is not None:
        return HeterographDataset(**heterograph_dataset_args)
    if args.dataset == 'moad':
        raise NotImplementedError
    
    pdb_args = model_conf_to_pdb_args(
        model_args,
        split_path=args.split_path,
        keep_original=True,
        require_ligand=True,
        transform=None,
        num_workers=1,
    )
    
    dataset = PDBBind(**pdb_args)

    return dataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default='workdir/test_score', help='Path to folder with trained score model and hyperparameters')
    parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use inside the folder')
    parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use inside the folder')
    parser.add_argument('--num_cpu', type=int, default=None, help='if this is a number instead of none, the max number of cpus used by torch will be set to this.')
    parser.add_argument('--run_name', type=str, default=None, help='')
    parser.add_argument('--project', type=str, default='ligbind_inf', help='')
    parser.add_argument('--batch_size', type=int, default=40, help='Number of poses to sample in parallel')
    parser.add_argument('--log_dir', type=str, default="workdir/out/random/", help='')

    parser.add_argument('--dataset', type=str, default='pdbbind', help='')
    parser.add_argument('--split_path', type=str, default='data/splits/timesplit_test', help='Path of file defining the split')

    parser.add_argument('--old_confidence_model', action='store_true', default=True, help='')
    parser.add_argument('--no_random', action='store_true', default=False, help='Whether to add randomness in diffusion steps')
    parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Whether to add noise after the final step')
    parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE')
    parser.add_argument('--wandb', action='store_true', default=False, help='') # TODO remove
    parser.add_argument('--inference_steps', type=int, default=40, help='Number of denoising steps')
    parser.add_argument('--limit_complexes', type=int, default=0, help='Limit to the number of complexes')
    parser.add_argument('--save_visualisation', action='store_true', default=True, help='Whether to save visualizations')
    parser.add_argument('--samples_per_complex', type=int, default=4, help='Number of poses to sample for each complex')
    parser.add_argument('--resample_rdkit', action='store_true', default=False, help='')
    parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='Schedule type, no other options')
    parser.add_argument('--inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')
    parser.add_argument('--inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')
    parser.add_argument('--pocket_knowledge', action='store_true', default=False, help='')
    parser.add_argument('--no_random_pocket', action='store_true', default=False, help='')
    parser.add_argument('--pocket_tr_max', type=float, default=3, help='')
    parser.add_argument('--pocket_cutoff', type=float, default=5, help='')
    parser.add_argument('--actual_steps', type=int, default=None, help='')
    parser.add_argument('--restrict_cpu', action='store_true', default=False, help='')
    parser.add_argument('--limit_failures', type=float, default=5, help='')
    parser.add_argument('--initial_noise_std_proportion', type=float, default=-1.0, help='Initial noise std proportion')
    parser.add_argument('--choose_residue', action='store_true', default=False, help='')

    parser.add_argument('--temp_sampling_tr', type=float, default=1.0)
    parser.add_argument('--temp_psi_tr', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_tr', type=float, default=0.5)
    parser.add_argument('--temp_sampling_rot', type=float, default=1.0)
    parser.add_argument('--temp_psi_rot', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_rot', type=float, default=0.5)
    parser.add_argument('--temp_sampling_tor', type=float, default=1.0)
    parser.add_argument('--temp_psi_tor', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_tor', type=float, default=0.5)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "rt") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value


    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run output will be stored in {out_dir}")

    # record parameters
    yaml_file_name = os.path.join(out_dir, 'model_parameters.yml')
    if not os.path.exists(yaml_file_name):
        save_yaml_file(yaml_file_name, args.__dict__)

    if args.restrict_cpu:
        threads = 16
        os.environ["OMP_NUM_THREADS"] = str(threads)  # export OMP_NUM_THREADS=4
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)  # export OPENBLAS_NUM_THREADS=4
        os.environ["MKL_NUM_THREADS"] = str(threads)  # export MKL_NUM_THREADS=6
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)  # export VECLIB_MAXIMUM_THREADS=4
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)  # export NUMEXPR_NUM_THREADS=6
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        torch.set_num_threads(threads)

    
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    if args.confidence_model_dir is not None:
        with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
            confidence_args = Namespace(**yaml.full_load(f))
        if not os.path.exists(confidence_args.original_model_dir):
            print("Path does not exist: ", confidence_args.original_model_dir)
            confidence_args.original_model_dir = os.path.join(*confidence_args.original_model_dir.split('/')[-2:])
            print('instead trying path: ', confidence_args.original_model_dir)

    if args.num_cpu is not None:
        torch.set_num_threads(args.num_cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device found: {device}")

    test_dataset = get_dataset(
        args,
        score_model_args,
        getattr(args, "score_model_heterograph_dataset_args"),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    if args.confidence_model_dir is not None:
        if not (confidence_args.use_original_model_cache or confidence_args.transfer_weights):
            # if the confidence model uses the same type of data as the original model then we do not need this dataset and can just use the complexes
            print('HAPPENING | confidence model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the confidence model now.')
            confidence_test_dataset = get_dataset(
                args,
                confidence_args,
                getattr(args, "confidence_model_heterograph_dataset_args"),
            )
            num_test_samples = confidence_test_dataset.len()
            confidence_complex_to_id_dict = {}
            for id_ in range(confidence_test_dataset.len()):
                if isinstance(confidence_test_dataset, PDBBind):
                    complex = confidence_test_dataset.get(id_)
                    complex_name = complex.name
                elif isinstance(confidence_test_dataset, HeterographDataset):
                    complex_name = confidence_test_dataset.get_complex_name(id_)
                confidence_complex_to_id_dict[complex_name] = id_

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
    if args.ckpt == 'last_model.pt':
        model_state_dict = state_dict['model']
        ema_weights_state = state_dict['ema_weights']
        model.load_state_dict(model_state_dict, strict=True)
        ema_weights = ExponentialMovingAverage(model.parameters(), decay=score_model_args.ema_rate)
        ema_weights.load_state_dict(ema_weights_state, device=device)
        ema_weights.copy_to(model.parameters())
    else:
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
    if args.confidence_model_dir is not None:
        if confidence_args.transfer_weights:
            raise NotImplementedError  # Not supported, exists in evaluate.py if needed
        confidence_model_args = confidence_args

        confidence_model = get_model(confidence_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                    confidence_mode=True, old=args.old_confidence_model)
        state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None
        confidence_args = None
        confidence_model_args = None

    if args.wandb:
        import wandb
        run = wandb.init(
            entity='',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )

    if args.pocket_knowledge and args.different_schedules:
        t_max = (np.log(args.pocket_tr_max) - np.log(score_model_args.tr_sigma_min)) / (
                    np.log(score_model_args.tr_sigma_max) - np.log(score_model_args.tr_sigma_min))
    else:
        t_max = 1

    tr_schedule = get_t_schedule(
        sigma_schedule=args.sigma_schedule,
        inference_steps=args.inference_steps,
        inf_sched_alpha=args.inf_sched_alpha,
        inf_sched_beta=args.inf_sched_beta,
        t_max=t_max
    )
    t_schedule = None
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    print('common t schedule', tr_schedule)

    failures, skipped, names_list = 0, 0, []
    run_times = []
    N = args.samples_per_complex
    print('Size of test dataset: ', len(test_dataset))

    complex_path = os.path.join(out_dir, "complexes_out")
    os.makedirs(complex_path, exist_ok=True)
    if args.save_visualisation:
        visualization_dir = os.path.join(out_dir, "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)

    for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
        complex_save_fp = os.path.join(
                complex_path,
                f"{orig_complex_graph.name[0]}.pkl"
            )
        if os.path.exists(complex_save_fp):
            continue
        if args.limit_complexes > 0 and (idx >= args.limit_complexes):
            print("Breaking as we are done analysing complexes")
            break
        if "cuda" in str(device):
            torch.cuda.empty_cache()

        if confidence_model is not None and not (confidence_args.use_original_model_cache or confidence_args.transfer_weights) \
                and orig_complex_graph.name[0] not in confidence_complex_to_id_dict.keys():
            skipped += 1
            print(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name[0]}. We are skipping this complex.")
            continue
        success = 0
        bs = args.batch_size
        while 0 >= success > -args.limit_failures:
            try:
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                if args.resample_rdkit:
                    for i, g in enumerate(data_list):
                        g['ligand'].pos = g['ligand'].pos[i]

                randomize_position(data_list, score_model_args.no_torsion, args.no_random or args.no_random_pocket,
                                   score_model_args.tr_sigma_max if not args.pocket_knowledge else args.pocket_tr_max,
                                   args.pocket_knowledge, args.pocket_cutoff,
                                   initial_noise_std_proportion=args.initial_noise_std_proportion,
                                   choose_residue=args.choose_residue)


                pdb = None
                if args.save_visualisation:
                    visualization_list = []
                    for idx, graph in enumerate(data_list):
                        lig = orig_complex_graph.mol[0]
                        pdb = PDBFile(lig)
                        pdb.add(lig, 0, 0)
                        pdb.add(((orig_complex_graph['ligand'].pos if not args.resample_rdkit else orig_complex_graph['ligand'].pos[idx]) + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                        pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                        visualization_list.append(pdb)
                else:
                    visualization_list = None

                start_time = time.time()
                if not args.no_model:
                    if confidence_model is not None and not (
                            confidence_args.use_original_model_cache or confidence_args.transfer_weights):
                        confidence_complex_id = confidence_complex_to_id_dict[orig_complex_graph.name[0]]
                        orig_confidence_complex_graph = confidence_test_dataset.get(confidence_complex_id)
                        confidence_data_list = [copy.deepcopy(orig_confidence_complex_graph) for _ in
                                               range(N)]
                    else:
                        confidence_data_list = None
                        orig_confidence_complex_graph = None
                    data_list, confidence = sampling(
                        data_list=data_list, model=model,
                        inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                        tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                        tor_schedule=tor_schedule,
                        device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                        no_random=args.no_random,
                        ode=args.ode, visualization_list=visualization_list,
                        confidence_model=confidence_model,
                        confidence_data_list=confidence_data_list,
                        confidence_model_args=confidence_model_args,
                        t_schedule=t_schedule,
                        batch_size=bs,
                        no_final_step_noise=args.no_final_step_noise, pivot=None,
                        temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot, args.temp_sampling_tor],
                        temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                        temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot, args.temp_sigma_data_tor],
                        mixed_precision_inference=args.mixed_precision_inference,
                    )
                    confidence_out = confidence

                run_times.append(time.time() - start_time)
                if score_model_args.no_torsion:
                    orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

                ligand_pos = np.array([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in data_list])
                data_to_dump = {
                    "predicted_ligand_pos": ligand_pos,
                    "confidence": confidence_out.cpu().numpy(),
                }
                if hasattr(orig_complex_graph, "orig_complex_graph_fp"):
                    data_to_dump["orig_complex_graph_fp"] = orig_complex_graph.complex_graph_fp[0]
                if hasattr(orig_complex_graph, "mol_fp"):
                    data_to_dump["orig_mol_fp"] = orig_complex_graph.mol_fp[0]
                if hasattr(orig_complex_graph, "ligand_graph_fp"):
                    data_to_dump["docking_model_ligand_graph_fp"] = orig_complex_graph.ligand_graph_fp[0]
                if hasattr(orig_complex_graph, "protein_graph_fp"):
                    data_to_dump["docking_model_protein_graph_fp"] = orig_complex_graph.protein_graph_fp[0]
                if orig_confidence_complex_graph is not None:
                    if hasattr(orig_confidence_complex_graph, "orig_conf_complex_graph_fp"):
                        data_to_dump["orig_conf_complex_graph_fp"] = orig_confidence_complex_graph.complex_graph_fp
                    if hasattr(orig_confidence_complex_graph, "mol_fp"):
                        data_to_dump["orig_conf_mol_fp"] = orig_confidence_complex_graph.mol_fp
                    if hasattr(orig_confidence_complex_graph, "ligand_graph_fp"):
                        data_to_dump["conf_model_ligand_graph_fp"] = orig_confidence_complex_graph.ligand_graph_fp
                    if hasattr(orig_confidence_complex_graph, "protein_graph_fp"):
                        data_to_dump["conf_model_protein_graph_fp"] = orig_confidence_complex_graph.protein_graph_fp
                with open(complex_save_fp, "wb") as f:
                    pickle.dump(
                        data_to_dump,
                        f
                    )

                if args.save_visualisation:
                    for batch_idx in range(len(data_list)):
                        visualization_list[batch_idx].write(
                            f'{visualization_dir}/{data_list[batch_idx]["name"][0]}_{batch_idx}.pdb'
                            )
                names_list.append(orig_complex_graph.name[0])
                success = 1
            except Exception as e:
                print(f"Failed on {orig_complex_graph['name']}, success={success}, error={e}")
                last_traceback = traceback.format_exc()
                success -= 1
                if bs > 1:
                    bs = bs // 2

        if success != 1:
            # TODO: shall we save something here?
            failures += 1
            print(f"Failed on {orig_complex_graph['name']}, no more retries. The last tracebak below.")
            print(last_traceback)

    print(failures, "failures due to exceptions")
    print(skipped, ' skipped because complex was not in confidence dataset')
