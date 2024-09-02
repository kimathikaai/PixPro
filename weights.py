import pdb

import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    "resnet50",
    encoder_weights="imagenet",
    classes=12,
    in_channels=3,
    activation=None,
    encoder_depth=5,
    decoder_channels=[256, 128, 64, 32, 16],
)

pretrain_path = "./output/pixpro_trial/current.pth"
pretrain = torch.load(pretrain_path)
print(f"{pretrain.keys() = }")
# print(f"{pretrain['model'].keys() = }")
state_dict = {
    x.replace("encoder.", ""): y
    for x, y in pretrain["model"].items()
    if "encoder." in x
}
print(model.encoder.load_state_dict(state_dict, strict=True))
pdb.set_trace()
