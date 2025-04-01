import numpy as np
import collections
import os

synthetic_scenes = ['0001'] # , '0003', '0004', '0005', '0080', '0083', '0086', '0090', '0214', '0224', '0231', '0235', '0326', '0327', '0331', '0335', '0007', '0008', '0012', '0013', '0115', '0117', '0122', '0130', '0240', '0243', '0252', '0258', '0341', '0348', '0360', '0377', '0016', '0022', '0023', '0015', '0137', '0141', '0147', '0148', '0056', '0057', '0063', '0064', '0026', '0027', '0032', '0035', '0149', '0160', '0200', '0205', '0269', '0281', '0294', '0299', '0407', '0411', '0446', '0472', '0496', '0065', '0067', '0071', '0095', '0098', '0102', '0306', '0307', '0312', '0493', '0036', '0042', '0058', '0061']

megadepth_data_dict = {}

existing_megadepth_root = '/mnt/slarge/dust3r_data/megadepth_processed_extra'

for scene_id in synthetic_scenes:
    scene_path = os.path.join(existing_megadepth_root, scene_id, '0')
    # lst all files ending with .jpg
    image_files = [f for f in os.listdir(scene_path) if f.endswith('.jpg')]
    # remove the .jpg extension (but it's .jpg.jpg, just want to remove the last .jpg)
    image_files = [f[:-4] for f in image_files]

    megadepth_data_dict[(scene_id, '0')] = image_files

# Write to npz file
np.savez('megadepth_image_list_v0.npz', data=megadepth_data_dict)

loaded = np.load('megadepth_image_list_v0.npz', allow_pickle=True)
megadepth_data_dict = loaded['data'].item()

print(megadepth_data_dict)

