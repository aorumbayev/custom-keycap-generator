from build123d import *
import argparse
import os
import yaml
from tqdm import tqdm
from key import KeyConfig, Key, export_stl, export_brep, export_step, Mesher
from stem import stem_from_config

parser = argparse.ArgumentParser(
    "Custom Keycap Generator",
    "Generate custom print-ready keycap geometries",
    "Work in progress. Contributions welcome."
)
parser.add_argument('style')
parser.add_argument('layout')
parser.add_argument('-o', '--output-path', default="output")
parser.add_argument('-f', '--format', default='stl', choices=['stl', 'brep', 'step', '3mf'])

if __name__ == '__main__':
    args = parser.parse_args()

    with open(f"configs/styles/{args.style}.yaml") as f:
        style = yaml.safe_load(f.read())
    with open(f"configs/layouts/{args.layout}.yaml") as f:
        layout = yaml.safe_load(f)

    print(f"Generating {len(layout['keys'])} keys...")
    for key_name, key_conf in tqdm(layout['keys'].items()):
        base = key_conf.pop('base', '')
        modifiers = key_conf.pop('modifiers', [])
        config = (
                style['global'] |
                style['bases'].get(base, {}) |
                key_conf
        )
        for mod in modifiers:
            config = config | style['modifiers'][mod]
        stem = stem_from_config(**config.pop('stem', {}))
        key_config = KeyConfig(**config)
        key = Key(key_config, stem)

        out_path = os.path.join(args.output_path, f"{key_name}.{args.format}")
        if args.format == 'stl':
            shape_to_export = key.shape(return_components=False)
            export_stl(shape_to_export, out_path)
        elif args.format == 'brep':
            shape_to_export = key.shape(return_components=False)
            export_brep(shape_to_export, out_path)
        elif args.format == 'step':
            shape_to_export = key.shape(return_components=False)
            export_step(shape_to_export, out_path)
        elif args.format == '3mf':
            key_body, legend_parts = key.shape(return_components=True)
            
            mesher = Mesher()
            mesher.add_shape(key_body)
            for legend_part in legend_parts:
                if legend_part:
                    mesher.add_shape(legend_part)
            mesher.write(out_path)
        else:
            print(f"Warning: Unsupported format '{args.format}' for key '{key_name}'")
