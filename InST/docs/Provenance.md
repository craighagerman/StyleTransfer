# Provenance

> The term “data provenance”, sometimes called “data lineage,” refers to a documented trail that accounts for the origin of a piece of data


Note
The InST repo appears to have copy/pasted code from other repos (e.g. Stable Diffusion) with some minor changes. The purpose of this file is to investigate (1) what is original to the InST authors and (2) what is not and (3) identify where the non-original code comes from. (I think proper SE practice would be to create separate modules and not copy into the same root directory. That is my aim.)

-----

InST = Authors of InST paper/code
- [InST | github](https://github.com/zyxElsa/InST)
SD = stable-diffusion repo
- [Stable-diffusion | github](https://github.com/CompVis/stable-diffusion)
LR = lucidrains repo
- [X-Transformers | github](https://github.com/lucidrains/x-transformers/tree/main/x_transformers)
DSD = Dreambooth Stable Diffusion
- [Dreambooth-Stable-Diffusion | github](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)


| Path | Author | InST edits | Notes |
| ---- | ---- | ---- | ---- |
| **./InST/configs:** | InST |  |  |
| autoencoder | SD |  |  |
| latent-diffusion | SD |  |  |
| stable-diffusion | SD |  |  |
|  |  |  |  |
| **./InST/configs/autoencoder:** | SD |  |  |
| autoencoder_kl_16x16x16.yaml | SD |  |  |
| autoencoder_kl_32x32x4.yaml | SD |  |  |
| autoencoder_kl_64x64x3.yaml | SD |  |  |
| autoencoder_kl_8x8x64.yaml | SD |  |  |
|  |  |  |  |
| **./InST/configs/latent-diffusion:** | SD |  |  |
| celebahq-ldm-vq-4.yaml | SD |  |  |
| cin-ldm-vq-f8.yaml | SD |  |  |
| cin256-v2.yaml | SD |  |  |
| ffhq-ldm-vq-4.yaml | SD |  |  |
| lsun_bedrooms-ldm-vq-4.yaml | SD |  |  |
| lsun_churches-ldm-kl-8.yaml | SD |  |  |
| txt2img-1p4B-eval.yaml | SD |  |  |
| txt2img-1p4B-eval_with_tokens.yaml | InST |  |  |
| txt2img-1p4B-finetune.yaml | InST |  |  |
| txt2img-1p4B-finetune_style.yaml | InST |  |  |
|  |  |  |  |
| **./InST/configs/stable-diffusion:** | SD |  |  |
| v1-finetune.yaml | InST |  |  |
| v1-inference.yaml | SD | Minor |  |
|  |  |  |  |
| **./InST/evaluation:** |  |  |  |
| clip_eval.py | InST |  |  |
|  |  |  |  |
| **./InST/ldm:** | InST |  |  |
| data | SD |  |  |
| lr_scheduler.py | SD |  |  |
| models | SD |  |  |
| modules | SD |  |  |
| util.py | SD |  |  |
|  |  |  |  |
| **./InST/ldm/data:** | SD |  |  |
| __init__.py | SD |  |  |
| base.py | SD |  |  |
| imagenet.py | SD |  |  |
| lsun.py | SD |  |  |
| personalized.py | InST |  |  |
|  |  |  |  |
| **./InST/ldm/models:** | SD |  |  |
| autoencoder.py | SD | minor |  |
| diffusion | SD |  |  |
|  |  |  |  |
| **./InST/ldm/models/diffusion:** | SD |  |  |
| __init__.py | SD |  |  |
| classifier.py | SD |  |  |
| ddim.py | SD | minor |  |
| ddpm.py | SD | major |  |
| plms.py | SD |  |  |
|  |  |  |  |
| **./InST/ldm/modules:** | SD |  |  |
| attention.py | SD |  |  |
| diffusionmodules | SD |  |  |
| distributions | SD |  |  |
| ema.py | SD |  |  |
| embedding_manager.py | SD |  |  |
| encoders | SD |  |  |
| image_degradation | SD |  |  |
| losses | SD |  |  |
| x_transformer.py | SD/LR | minor | origin: lucidrains |
|  |  |  |  |
| **./InST/ldm/modules/diffusionmodules:** | SD |  |  |
| __init__.py | SD |  |  |
| model.py | SD |  |  |
| openaimodel.py | SD |  |  |
| util.py | SD | minor |  |
|  |  |  |  |
| **./InST/ldm/modules/distributions:** | SD |  |  |
| __init__.py | SD |  |  |
| distributions.py | SD |  |  |
|  |  |  |  |
| **./InST/ldm/modules/encoders:** | SD |  |  |
| __init__.py | SD |  |  |
| modules.py | SD | major |  |
| modules_bak.py | InST |  |  |
|  |  |  |  |
| **./InST/ldm/modules/image_degradation:** | SD |  |  |
| __init__.py | SD |  |  |
| bsrgan.py | SD |  |  |
| bsrgan_light.py | SD |  |  |
| utils | SD |  |  |
| utils_image.py | SD |  |  |
|  |  |  |  |
| **./InST/ldm/modules/image_degradation/utils**: | SD |  |  |
| test.png | SD |  |  |
|  |  |  |  |
| **./InST/ldm/modules/losses:** | SD |  |  |
| __init__.py | SD |  |  |
| contperceptual.py | SD |  |  |
| vqperceptual.py | SD |  |  |
|  |  |  |  |
| **./InST/models:** | SD |  |  |
| first_stage_models | SD |  |  |
| ldm | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models:** | SD |  |  |
| kl-f16 | SD |  |  |
| kl-f32 | SD |  |  |
| kl-f4 | SD |  |  |
| kl-f8 | SD |  |  |
| vq-f16 | SD |  |  |
| vq-f4 | SD |  |  |
| vq-f4-noattn | SD |  |  |
| vq-f8 | SD |  |  |
| vq-f8-n256 | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models/kl-f16:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models/kl-f32:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models/kl-f4:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models/kl-f8:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models/vq-f16:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models/vq-f4:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models/vq-f4-noattn:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models/vq-f8:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/first_stage_models/vq-f8-n256**: | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm:** | SD |  |  |
| bsr_sr | SD |  |  |
| celeba256 | SD |  |  |
| cin256 | SD |  |  |
| ffhq256 | SD |  |  |
| inpainting_big | SD |  |  |
| layout2img-openimages256 | SD |  |  |
| lsun_beds256 | SD |  |  |
| lsun_churches256 | SD |  |  |
| semantic_synthesis256 | SD |  |  |
| semantic_synthesis512 | SD |  |  |
| text2img256 | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/bsr_sr:** | SD |  |  |
| config.yaml | SD |  |  |
|  | SD |  |  |
| **./InST/models/ldm/celeba256:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/cin256:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/ffhq256:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/inpainting_big:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/layout2img-openimages256:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/lsun_beds256:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/lsun_churches256:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/semantic_synthesis256:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/semantic_synthesis512:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/models/ldm/text2img256:** | SD |  |  |
| config.yaml | SD |  |  |
|  |  |  |  |
| **./InST/scripts:** | SD |  |  |
| download_first_stages.sh | SD |  |  |
| download_models.sh | SD |  |  |
| evaluate_model.py | InST |  |  |
| inpaint.py | SD |  |  |
| latent_imagenet_diffusion.ipynb | SD |  |  |
| sample_diffusion.py | SD |  |  |
| stable_txt2img.py | DSD? | minor |  |
| stable_txt2style.py | SD? |  |  |
| txt2img.py | SD | major |  |
|  |  |  |  |




