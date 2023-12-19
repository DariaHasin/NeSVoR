# from argparse import Namespace
# from typing import List
# import torch
# from ..transform import transform_points
# from ..image import Slice, Volume
# from .models import INR
# from ..utils import resolution2sigma, meshgrid
# import pdb
# import os


# def sample_volume(model: INR, mask: Volume, args: Namespace) -> Volume:
#     model.eval()
#     img = mask.resample(args.output_resolution, None)
#     img.image[img.mask] = sample_points(model, img.xyz_masked, args)
#     return img


# def sample_points(model: INR, xyz: torch.Tensor, args: Namespace) -> torch.Tensor:
#     shape = xyz.shape[:-1]
#     xyz = xyz.view(-1, 3)
#     v = torch.empty(xyz.shape[0], dtype=torch.float32, device=args.device)
#     batch_size = args.inference_batch_size
#     with torch.no_grad():
#         for i in range(0, xyz.shape[0], batch_size):
#             xyz_batch = xyz[i : i + batch_size]
#             xyz_batch = model.sample_batch(
#                 xyz_batch,
#                 None,
#                 resolution2sigma(args.output_resolution, isotropic=True),
#                 args.n_inference_samples if args.output_psf else 0,
#             )
#             v_b = model(xyz_batch, False).mean(-1)
#             v[i : i + batch_size] = v_b
#     return v.view(shape)


# def sample_slice(model: INR, slice: Slice, mask: Volume, args: Namespace) -> Slice:
#     # clone the slice
#     slice_sampled = slice.clone()
#     slice_sampled.image = torch.zeros_like(slice_sampled.image)
#     slice_sampled.mask = torch.zeros_like(slice_sampled.mask)
#     xyz = meshgrid(slice_sampled.shape_xyz, slice_sampled.resolution_xyz).view(-1, 3)
#     m = mask.sample_points(transform_points(slice_sampled.transformation, xyz)) > 0
#     if m.any():
#         xyz_masked = model.sample_batch(
#             xyz[m],
#             slice_sampled.transformation,
#             resolution2sigma(slice_sampled.resolution_xyz, isotropic=False),
#             args.n_inference_samples if args.output_psf else 0,
#         )
#         v = model(xyz_masked, False).mean(-1)
#         slice_sampled.mask = m.view(slice_sampled.mask.shape)
#         slice_sampled.image[slice_sampled.mask] = v.to(slice_sampled.image.dtype)
#     return slice_sampled


# def sample_slices(
#     model: INR, slices: List[Slice], mask: Volume, args: Namespace
# ) -> List[Slice]:
#     model.eval()
#     with torch.no_grad():
#         slices_sampled = []
#         for i, slice in enumerate(slices):
#             torch.cuda.empty_cache()
#             print('i is: ', i)
#             slices_sampled = sample_slice(model, slice, mask, args)
#             slices_sampled.save(os.path.join(args.simulated_slices, f"{i}.nii.gz"), True)

#     return slices_sampled

from argparse import Namespace
from typing import List
import torch
from ..transform import transform_points
from ..image import Slice, Volume
from .models import INR
import torch.nn.functional as F
from ..utils import resolution2sigma, meshgrid
import os
import pdb


def sample_volume(model: INR, mask: Volume, args: Namespace) -> Volume:
    model.eval()
    img = mask.resample(args.output_resolution, None)
    img.image[img.mask] =  sample_points(model.inr, img.xyz_masked, args)
    return img


def sample_points(model: INR, xyz: torch.Tensor, args: Namespace) -> torch.Tensor:
    shape = xyz.shape[:-1]
    xyz = xyz.view(-1, 3)
    v = torch.empty(xyz.shape[0], dtype=torch.float32, device=args.device)
    batch_size = args.inference_batch_size
    with torch.no_grad():
        for i in range(0, xyz.shape[0], batch_size):
            xyz_batch = xyz[i : i + batch_size]
            xyz_batch = model.sample_batch(
                xyz_batch,
                None,
                resolution2sigma(args.output_resolution, isotropic=True),
                args.n_inference_samples if args.output_psf else 0,
            )
            v_b = model(xyz_batch, False).mean(-1)
            v[i : i + batch_size] = v_b
    return v.view(shape)


def sample_slice(model: INR, slice: Slice, mask: Volume, args: Namespace) -> Slice:
    # clone the slice
    slice_sampled = slice.clone()
    slice_sampled.image = torch.zeros_like(slice_sampled.image)
    slice_sampled.mask = torch.zeros_like(slice_sampled.mask)
    xyz = meshgrid(slice_sampled.shape_xyz, slice_sampled.resolution_xyz).view(-1, 3)
    m = mask.sample_points(transform_points(slice_sampled.transformation, xyz)) > 0
    if m.any():
        xyz_masked = model.inr.sample_batch(
            xyz[m],
            slice_sampled.transformation,
            resolution2sigma(slice_sampled.resolution_xyz, isotropic=False),
            args.n_inference_samples if args.output_psf else 0,
        )
        breakpoint()
        v = model.inr(xyz_masked, False).mean(-1)
        slice_sampled.mask = m.view(slice_sampled.mask.shape)
        slice_sampled.image[slice_sampled.mask] = v.to(slice_sampled.image.dtype)
    return slice_sampled


def sample_slice_var(model, slice_i, mask, i ,args):
  slice_idx =  torch.tensor([i],device = args.device)
  model.eval()
  with torch.no_grad():
    slice_sampled = slice_i.clone()
    slice_sampled.image = torch.zeros_like(slice_sampled.image)
    slice_sampled.mask = torch.zeros_like(slice_sampled.mask)
    xyz = meshgrid(slice_sampled.shape_xyz, slice_sampled.resolution_xyz).view(-1, 3)
    m = mask.sample_points(transform_points(slice_sampled.transformation, xyz)) > 0
    if m.any():
      xyz_masked = model.inr.sample_batch(
          xyz[m],
          slice_sampled.transformation,
          resolution2sigma(slice_sampled.resolution_xyz, isotropic=False),
          args.n_inference_samples if args.output_psf else 0,
          # 0 if args.no_output_psf else args.n_inference_samples,
      )
      v, pe, z = model.inr(xyz_masked, True)
      prefix_shape = v.shape
      se = model.slice_embedding(slice_idx)[:, None].expand(-1, z.shape[0], -1)  

      zs = [] 
      zs.append(se.reshape(-1, se.shape[-1]))      

      zs.append(z[..., 1:])
      var = model.sigma_net(torch.cat(zs, -1))
      var = F.softplus(var.view(prefix_shape)).mean(-1)
      var = var.log()
      slice_sampled = slice_i.clone()
      slice_sampled.mask = m.view(slice_sampled.mask.shape)
      slice_sampled.image[slice_sampled.mask] = var.to(slice_sampled.image.dtype)
    return slice_sampled


def sample_slices(
    model: INR, slices: List[Slice], mask: Volume, args: Namespace
) -> List[Slice]:
    model.eval()
    with torch.no_grad():
        slices_sampled = []
        for i, slice in enumerate(slices):
            torch.cuda.empty_cache()
            print('i is: ', i)
            # slices_sampled.append(sample_slice(model, slice, mask, args))
            try:
              slices_sampled = sample_slice_var(model, slice, mask, i, args)
            except:
              print("sample_slice_var FAILED")
              break
            slices_sampled.save(os.path.join(args.simulated_slices, f"{i}.nii.gz"), True)
    return slices_sampled
