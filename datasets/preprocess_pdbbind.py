import argparse
import yaml

from .pdbbind import model_conf_to_pdb_args, PDBBind


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument("--split_path", type=str, default="")
    parser.add_argument("--keep_original", type=bool, default=True)
    parser.add_argument("--require_ligand", type=bool, default=False)
    parser.add_argument("--transform", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--force_regenerate", action="store_true", default=False)
    parser.add_argument("--csv_lines_to_process", type=int, default=None)

    args = parser.parse_args()
    
    if args.transform is not None:
        raise NotImplementedError
    
    with open(args.config, "rb") as f:
        loaded_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    model_conf = argparse.Namespace(**loaded_config_dict)

    pdb_args = model_conf_to_pdb_args(
        model_conf_args=model_conf,
        split_path=args.split_path,
        keep_original=args.keep_original,
        require_ligand=args.require_ligand,
        transform=args.transform,
        num_workers=args.num_workers,
        force_regenerate=args.force_regenerate,
        csv_lines_to_process=args.csv_lines_to_process,
    )

    print("Starting generating PDBBind dataset with arguments:", pdb_args)
    pdbbind_dataset = PDBBind(**pdb_args)

    print(f"Done, dataset exists at: {pdbbind_dataset.full_cache_path}")