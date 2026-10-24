import torch

from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS


@MODELS.register_module()
class CustomDataPreprocessor(DetDataPreprocessor):

    def forward(self, data, training=False):
        data = self.cast_data(data)

        inputs = data["inputs"]
        assert isinstance(inputs, dict), (
            "Expected inputs to be a dict containing 'img' "
            "and the voxel projection tensors."
        )
        assert "img" in inputs

        img = inputs["img"]
        
        voxel_proj_idx = inputs.get("voxel_proj_index")
        voxel_valid_mask = inputs.get("voxel_valid_mask")

        if isinstance(voxel_proj_idx, (list, tuple)):
            voxel_proj_idx = torch.stack(voxel_proj_idx, dim=0)

        if isinstance(voxel_valid_mask, (list, tuple)):
            voxel_valid_mask = torch.stack(voxel_valid_mask, dim=0)

        processed = super().forward(
            {
                "inputs": img,
                "data_samples": data.get("data_samples"),
            },
            training=training,
        )

        processed_inputs = {
            "img": processed["inputs"],
        }

        if voxel_proj_idx is not None:
            processed_inputs["voxel_proj_index"] = voxel_proj_idx

        if voxel_valid_mask is not None:
            processed_inputs["voxel_valid_mask"] = voxel_valid_mask

        processed["inputs"] = processed_inputs

        return processed