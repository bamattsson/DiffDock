import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
import traceback
from types import SimpleNamespace
from pathlib import Path
import copy

import yaml
import pickle
import torch
import tqdm
from torch_geometric.data import HeteroData

from utils.utils import get_model, save_yaml_file
from datasets.pdbbind import PDBBind, model_conf_to_pdb_args
from utils.sampling import compute_confidence


def get_dataset(
        args: SimpleNamespace,
        model_args: Namespace,
    ) -> PDBBind:
    if args.dataset != 'pdbbind':
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


class ConfidenceInferenceDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            confidence_model_heterographs: PDBBind,
            infer_path: Path,
    ):
        self.confidence_model_heterographs = confidence_model_heterographs
        self.infer_path = infer_path

        # Create map to be able to access confidence model heterographs
        self.confidence_complex_to_id_dict = {}
        for id_ in tqdm.tqdm(
            range(self.confidence_model_heterographs.len()),
            desc="Indexing heterographs"
        ):
            complex = self.confidence_model_heterographs.get(id_)
            self.confidence_complex_to_id_dict[complex.name] = id_

        # Create map to be able to access complexes over which to infer
        self.id_to_complex_name_dict = {}
        for id_, path in enumerate(self.infer_path.glob("*.pkl")):
            complex_name = path.stem
            self.id_to_complex_name_dict[id_] = complex_name

    def __getitem__(self, index: int):
        complex_name = self.id_to_complex_name_dict[index]

        with open(self.infer_path.joinpath(f"{complex_name}.pkl"), "rb") as f:
            infer_data = pickle.load(f)
        
        if complex_name not in self.confidence_complex_to_id_dict:
            print(f"WARNING | Could not find heterograph for {complex_name}")
            return (complex_name, None)
        
        conf_model_heterograph = self.confidence_model_heterographs.get(
            self.confidence_complex_to_id_dict[complex_name]
        )

        # Some basic assertions to ensure the data is not corrupted
        orig_receptor_position = infer_data["orig_complex_graph"]["receptor"]["pos"]
        assert (
            orig_receptor_position == conf_model_heterograph["receptor"]["pos"]
        ).all().item(), f"Receptor position not preserved for {complex_name}"
        lig_atom_predictions = infer_data["predicted_ligand_pos"].shape[1]
        conf_model_num_lig_atoms = conf_model_heterograph["ligand"]["pos"].shape[0]
        assert (
            lig_atom_predictions == conf_model_num_lig_atoms
        ), f"Number of ligand atoms not identical for {complex_name}"

        # Prepare data for inference
        confidence_data_list = []
        num_predictions = infer_data["predicted_ligand_pos"].shape[0]
        for i in range(num_predictions):
            predicted_heterograph = copy.deepcopy(conf_model_heterograph)
            predicted_heterograph["ligand"].pos = torch.tensor(
                infer_data["predicted_ligand_pos"][i])
            confidence_data_list.append(predicted_heterograph)
        
        return (complex_name, confidence_data_list)
    
    def __len__(self):
        return len(self.id_to_complex_name_dict)


def inactive_collate_fn(
    batch,
):
    return batch


def main(
        configs: SimpleNamespace
    ):

    # Load configs
    if configs.run_name is None:
        configs.run_name = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_dir = os.path.join(configs.log_dir, configs.run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run output will be stored in {out_dir}")

    # record parameters
    yaml_file_name = os.path.join(out_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, configs.__dict__)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device found: {device}")

    # Load model
    with open(f'{configs.confidence_model_dir}/model_parameters.yml') as f:
        confidence_model_args = Namespace(**yaml.full_load(f))
    if not os.path.exists(confidence_model_args.original_model_dir):
        raise ValueError(f"Path does not exist: {confidence_model_args.original_model_dir}")

    confidence_model = get_model(
        confidence_model_args,
        device,
        t_to_sigma=None,  # Not needed for confidence model
        no_parallel=True,
        confidence_mode=True,
        old=configs.old_confidence_model
    )
    state_dict = torch.load(
        f'{configs.confidence_model_dir}/{configs.confidence_ckpt}',
        map_location=torch.device('cpu'))
    if "model" in state_dict:
        state_dict = state_dict["model"]
    confidence_model.load_state_dict(state_dict, strict=True)
    # TODO: Need to add support for mixed precision models
    confidence_model = confidence_model.to(device)
    confidence_model.eval()

    # Load inference data set
    infer_path = Path(configs.complexes_to_infer_folder).joinpath("complexes_out")
    assert infer_path.exists, "infer_path needs to exist"
    confidence_inference_dataset = get_dataset(configs, confidence_model_args)
    confidence_inference_dataset = ConfidenceInferenceDataset(
        confidence_inference_dataset,
        infer_path,
    )
    conf_inference_loader = torch.utils.data.DataLoader(
        dataset=confidence_inference_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=inactive_collate_fn,
    )

    failures = 0
    print('Size of test dataset: ', len(confidence_inference_dataset))

    complex_path = os.path.join(out_dir, "complexes_out")
    os.makedirs(complex_path)

    for inference_data_sample in tqdm.tqdm(
        conf_inference_loader,
        desc="Predicting",
        total=len(confidence_inference_dataset),
        ):
        if "cuda" in str(device):
            torch.cuda.empty_cache()

        assert len(inference_data_sample) == 1, "batch_size needs to be 1"
        complex_name = inference_data_sample[0][0]
        heterograph_list = inference_data_sample[0][1]

        if heterograph_list is None:
            print(f"|WARNING: data did not contain {complex_name}")
            # TODO: shall we save something here?

        success = 0
        bs = configs.batch_size
        while 0 >= success > -configs.limit_failures:
            try:
                confidence = compute_confidence(
                    heterograph_list,
                    bs,
                    device,
                    confidence_model,
                    confidence_model_args,
                )
            
                complex_save_fp = os.path.join(
                    complex_path,
                    f"{complex_name}.pkl"
                )
                ligand_poses = [h["ligand"].pos.numpy() for h in heterograph_list]
                data_to_dump = {
                    "predicted_ligand_pos": ligand_poses,
                    "confidence": confidence.numpy(),
                }
                with open(complex_save_fp, "wb") as f:
                    pickle.dump(
                        data_to_dump,
                        f
                    )
                
                success = 1
            except Exception as e:
                print("Failed on", complex_name, e)
                print(traceback.format_exc())
                success -= 1
                if bs > 1:
                    bs = bs // 2
        if success != 1:
            # TODO: shall we save something here?
            failures += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)

    args = parser.parse_args()
    with open(args.config, "rt") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    configs = SimpleNamespace(**config_dict)
    main(configs=configs)