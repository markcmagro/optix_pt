# OptiX Path Tracer

The objective of this path tracer is to implement glTF 2.0's metallic-roughness model, which is more or less equivalent to Blender's Principled BSDF or Disney's BSDF.

A Monte Carlo approach is used for the layers in this model and material blending (not physically based) is supported.

Scenes are specified using glTF 2.0, typically produced by exporting from Blender 4.1.

The project is very much a work in progress, currently focussing on correct implementation rather than optimisation. This means that the code is unpolished, littered with OptiX printf statements and largely undocumented. The features currently implemented are the following:

- Metallic maps
- Roughness maps
- Normal maps
- Textures
- Diffuse
- Specular
- Microfacet BSDF (Trowbridge-Reitz / GGX) (reflection and transmission)
- Alpha blending
- Emissive geometry
- Point lights
- Progressive rendering
- Denoising (OptiX)
- Tone mapping and gamma correction
- Screenshots (PNG, Radiance RGBE, EXR) (press 1, 2 or 3 respectively)
- Real-time camera movement (WASD + Mouse)

Upcoming features:
- Sheen BRDF
- Environment lighting

Refer to my portfolio page (https://markcmagro.github.io/) for some renders produced by this path tracer (all images except for the first four, which were produced with a naive path tracer).

## Installation

1. Install CUDA 12.3  
   Create an environment variable called CUDA_PATH pointing to the install directory.  
   E.g. CUDA_PATH C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3  
   Note: This environment variable may be created automatically during the installation.

1. Install OptiX 7.1  
   Create an environment variable called OPTIX_PATH_7_1_0 pointing to the install directory.  
   E.g. OPTIX_PATH_7_1_0 D:\OptiX\OptiX SDK 7.1.0  
   Note: Install to a location with full read/write permissions.

1. Clone OpenEXR 3.2.2 from https://github.com/AcademySoftwareFoundation/openexr.git, build and install it.  
   Create an environment variable called OpenEXR_DIR pointing to the install directory.  
   E.g. OpenEXR_DIR D:\online-code\openexr-3.2.2\install
      
1. Clone the scenes repo from https://github.com/markcmagro/scenes.git.  
   Create an environment variable called SCENES_DIR pointing to the repo.  
   E.g. SCENES_DIR D:\scenes

1. Clone the optix_pt repo.  
   git clone --recurse-submodules git@github.com:markcmagro/optix_pt.git  
   If building with Visual Studio Community 2022 (recommended), set the startup item to optix_pt.exe.
