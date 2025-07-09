# RaySplats: Ray Tracing based Gaussian Splatting
Krzysztof Byrski, Marcin Mazur, Jacek Tabor, Tadeusz Dziarmaga, Marcin Kądziołka, Dawid Baran, Przemysław Spurek <br>

| arXiv |
| :---- |
| RaySplats: Ray Tracing based Gaussian Splatting [https://arxiv.org/pdf/2501.19196.pdf](http://arxiv.org/abs/2501.19196)|

<img src=assets/gif1.gif height="300" class="center"> 
<br>

<table align="center" cellspacing="0" cellpadding="0">
  <tr class="center">
    <td><img src=assets/screenshot1.png height="200" width="300" class="center"></td>
    <td><img src=assets/screenshot92.png height="200" width="300" class="center"></td>
    <td><img src=assets/screenshot10.png height="200" width="300" class="center"> </td>
  </tr>
  </tr class="center">
  <tr class="center">
    <td><img src=assets/screenshot7.png height="200" width="300" ></td>
    <td><img src=assets/screenshot82.png height="200" width="300" ></td>
    <td><img src=assets/screenshot4.png height="200" width="300" class="center"> </td>
  </tr>
</table>

# Features
- Spherical harmonics support up to the degree **4**.
- Interactive Windows viewer / optimizer application allowing to preview the trained model state in real time.
- Highly efficient pure Gaussian renderer (no embedding primitive mesh approximation).
- Highly configurable optimizer based on the convenient configuration file.
- Support for both the **Blender** and **COLMAP** data sets (after some preprocessing by the 3DGS).

## Controls in the interactive Windows viewer / optimizer application

- **Double Left Click**: Toggle between the **static camera** and the **free roam** mode.
- **Mouse Movement**: Rotate the camera in the **free roam** mode.
- **W / S**: Move forward / backward.
- **A / D**: Step left / right.
- **Spacebar / C**: Move up / down.
- **[ / ]**: Switch the camera to the previous / next rendering pose.

# RaySplatting Viewer
![image](https://github.com/user-attachments/assets/9a9d61cb-f54a-4301-8a52-4c2d0ce2cc72)
![](assets/tutorial.mp4)

This is a lightweight and user-friendly viewer for visualizing **RaySplatting** with additional user-loaded objects that support ray tracing. The viewer allows seamless integration of **OBJ** and **PLY (ASCII format)** files into the scene.  

The current material system is optimized for models designed to be **reflective** or **glass-like**, making it ideal for rendering high-quality visuals with realistic light interactions.  

## System Requirements  

To use this viewer, ensure your system meets the following requirements:  

- **Operating System**: Windows  
- **GPU**: NVIDIA RTX 20xx series or higher (**RTX 30xx+ recommended**)  
- **CUDA Version**: 12.4 or later  
- **Required DLLs**: Place the following files in the directory:  
  ```plaintext
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
  ```
  - `cudart64_12.dll`  
  - `cufft64_11.dll`  

## Installation & Usage  

1. Download the provided **ZIP file**.  [Download ZIP](https://drive.google.com/file/d/1XPivZb6-dVtuwQ3T9UrxOF2gTTnerGhp/view?usp=sharing)
2. Extract the contents.  
3. Run the **exe file**—no additional setup required!  
4. Modify mesh properties in **mesh_config.txt**.  
5. Change the base scene by editing the **PLY file path** in `config.txt`.  

## Controls  

- Exactly the same as in the interactive Windows viewer / optimizer application.

## Future Features  

We are actively developing new features, including:  
- **Enhanced mesh transformations** (scaling, rotation, position editing beyond `mesh_config.txt`)  
- **Screenshot capture** for rendered scenes  
- **View presets** to allow seamless switching between different perspectives  
- **And much more!**  

Stay tuned for updates and improvements!

# Learning

1. Prerequisites:
-----------------
- Install Visual Studio 2019 Enterprise;
- Install CUDA Toolkit 12.4.1;
- Install NVIDIA OptiX SDK 8.0.0;

2. Compiling the CUDA static library:
------------------------------------
- Create the new CUDA 12.4 Runtime project and name it "RaySplattingCUDA";
- Remove the newly created kernel.cu file with the code template;
- Add all the files from the directory "RaySplattingCUDA" to the project;
- Change project's Configuration to "Release, x64";
- Add OptiX "include" directory path to the project's Include Directories. On our test system, we had to add the following path:

"C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0\include"

- In Properties -> Configuration Properties -> CUDA C/C++ -> Device -> Code Generation type the compute capability and microarchitecture version of your GPU. On our test system with RTX 4070 GPU we added "compute_89,sm_89";
- In Properties -> Configuration Properties -> General -> Configuration Type select "Static library (.lib)";
- For files: "shaders.cu" and "shadersMesh.cu" in Properties -> Configuration Properties -> CUDA C/C++ change the suffix of Compiler Output (obj/cubin) from ".obj" to ".ptx";
- For files: "shaders.cu" and "shadersMesh.cu" in Properties -> Configuration Properties -> CUDA C/C++ -> NVCC Compilation Type select "Generate device-only .ptx file (-ptx)";
- Make the following changes in the file kernel2.cu specifying the location of the compiled *.ptx shader files:

Line 192:
FILE *f = fopen("<location of the compiled *.ptx shader files>/shaders.cu.ptx", "rb");

Line 201:
f = fopen("<location of the compiled *.ptx shader files>/shaders.cu.ptx", "rb");

Line 4340:
FILE *f = fopen("<location of the compiled *.ptx shader files>/shadersMesh.cu.ptx", "rb");

Line 4349:
f = fopen("<location of the compiled *.ptx shader files>/shadersMesh.cu.ptx", "rb");

On our test system, we used the following paths as the string literal passed to the fopen function:

"C:/Users/\<Windows username>/source/repos/RaySplattingCUDA/RaySplattingCUDA/x64/Release/shaders.cu.ptx"
<br>
"C:/Users/\<Windows username>/source/repos/RaySplattingCUDA/RaySplattingCUDA/x64/Release/shadersMesh.cu.ptx"

- Build the project;

3. Compiling the Windows interactive optimizer application:
-----------------------------------------------------------
- Create the new Windows Desktop Application project and name it "RaySplattingWindows";
- Remove the newly generated RaySplattingWindows.cpp file with the code template;
- Add all the files from the directory "RaySplattingWindows" to the project;
- Change project's Configuration to "Release, x64";
- In Properties -> Configuration Properties -> Linker -> Input -> Additional Dependencies add new lines:

"RaySplattingCUDA.lib" <br>
"cuda.lib" <br>
"cudart.lib" <br>
"cufft.lib" <br>

- In Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories add the "lib\x64" path of your CUDA toolkit. On our test system, we had to add the following path:

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64"

- In Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories add the path of the directory containing your compiled CUDA static library. On our test system, we had to add the following path:

"C:\Users\\\<Windows username>\source\repos\RaySplattingCUDA\x64\Release"

4. Training the first model:
----------------------------
- Create the directory "dump" in the main RaySplattingWindows project's directory and then create the subdirectory "dump\save" in the main RaySplattingWindows project's directory. The application will store the checkpoints here. On our test system we created those directories in the following directory:

"C:\Users\\\<Windows username>\source\repos\RaySplattingWindows\RaySplattingWindows"

- Train the model with 3DGS for some small number of epochs (for example 100) on some dataset (for example: "truck" from "Tanks and Temples");
- Copy the output file cameras.json to the dataset main directory;
- Convert all of the files in the subdirectory "images" located in the dataset main directory to 24-bit *.bmp file format without changing their names;
- Copy the configuration file "config.txt" to the project's main directory. On our test system we copied it to the following directory:

"C:\Users\\\<Windows username>\source\repos\RaySplattingWindows\RaySplattingWindows"

- In lines: 2 and 3 of the configuration file specify the location of the dataset main directory and the output 3DGS *.ply file obtained after short model pretraining (Important! With --sh_degree 0 as RaySplats uses the RGB model);
- Run the "RaySplattingWindows" project from the Visual Studio IDE;



