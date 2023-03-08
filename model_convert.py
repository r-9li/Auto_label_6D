import trimesh
import os

p = {
    # Folder containing the BOP datasets.
    'dataset_path': '/media/r/T7 Shield/converted_model',
}

models_path = os.path.join(p['dataset_path'], 'models')
for root, dirs, files in os.walk(models_path):
    for file in files:
        if file.endswith('ply'):
            path = os.path.join(root, file)
            print(path)
            mesh = trimesh.load(path)
            _ = mesh.apply_obb()
            mesh.export(file_obj=path)
