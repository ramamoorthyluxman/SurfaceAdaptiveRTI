# SurfaceAdaptiveRTI

## Introduction
SurfaceAdaptiveRTI is a Blender plugin built to be useful for developing surface adaptive acquisition methods with regards to RTI.

Currently the plugin is developed for Blender version 3.1 and uses only the cycles renderer.


## Installation
To install the plugin go to `file -> User Preferences… -> Add-ons -> install Add-on from File…` choose the file `__init.py__` and press on `Install Add-on from File…` . Once installed it need to be activated.  The plugin will be available on the 3DView on the Tool tab.

## Usage
![plugin](https://github.com/giach68/SyntheticRTI/blob/master/Documentation/plugin_full.png)

The plugin is divided in 4 panels:
- **Create**: its mainly purpose is to create lamps, cameras and to manage the material parameters we want to iterate over the combinations;
- **Render**: it prepares the environment for rendering;
- **Tools**: various tools to help building the set;
- **Debug**: various information about the scene.

## Create
**Light file**: here you can insert the filepath of the .lp file with the position of lamps. Using the folder button it is possible to use the file select mode.  
The .lp files are structured this way:  


