# SurfaceAdaptiveRTI

## Introduction
SurfaceAdaptiveRTI is a Blender plugin built to be useful for developing surface adaptive acquisition methods with regards to RTI.

Currently the plugin is developed for Blender version 3.1 and uses only the cycles renderer.


## Installation
To install the plugin go to `file -> User Preferences… -> Add-ons -> install Add-on from File…` choose the file `__init.py__` and press on `Install Add-on from File…` . Once installed it need to be activated.  The plugin will be available on the 3DView on the Tool tab.

## Usage
![plugin](https://github.com/ramamoorthyluxman/SurfaceAdaptiveRTI/blob/main/display.jpg)

The plugin is divided in 4 panels:
- **Surface**: You can upload .OBJ file and texture image directly. If your 3D file is of some other format, you can import it from file->import and ignore the surface option in the plugin
- **Lights**: If NBLP enabled, the plugin automatically configures the light positions on the go adapting to any imported surface. if disabled, .lp file can be uploaded to create lights
- **Camera**: Create a camera and adjust its field of view and focus. 
- **Acquisition**: Control the acquisition process



