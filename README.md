**Enhancing Earth Observation Data for Land Use and Land Cover Analysis using SRGAN Network**

**📌 Project Overview**
This project focuses on enhancing the quality of Earth Observation (EO) satellite imagery using the Super-Resolution Generative Adversarial Network (SRGAN) for improved Land Use and Land Cover (LULC) analysis.
Low-resolution satellite images often lose important spatial information, making accurate land classification difficult. To overcome this issue, SRGAN is used to generate high-resolution satellite imagery from low-resolution inputs. The enhanced images are then utilized for LULC mapping and analysis.
The project mainly studies satellite imagery of Hyderabad region and demonstrates how super-resolution techniques improve classification accuracy and visual quality for remote sensing applications.

**Features**
> Super-resolution enhancement using SRGAN
> Generator and Discriminator deep learning architecture
> Satellite image preprocessing and patch extraction
> Low Resolution (LR), High Resolution (HR), and Super-Resolved (SR) image generation
> LULC map generation using QGIS
> Performance evaluation using:
> PSNR
> SSIM
> Remote sensing and Earth observation analysis

**🛰️ Dataset**
The dataset used in this project was collected from the Copernicus Earth Observation Program.
The dataset contains:
> Low Resolution (LR) Images
> High Resolution (HR) Images
> Super Resolved (SR) Images
The satellite images include:
> Urban regions
> Vegetation
> Water bodies
> Agricultural lands
> Barren land
Preprocessing steps:
> Image resizing
> Normalization
> Downsampling
> Patch extraction
> Data augmentation

**🧠 SRGAN Architecture**
Generator Network
The Generator network enhances low-resolution images using:
Residual Blocks
Pixel Shuffle Layers
Deconvolution Layers
Discriminator Network
The Discriminator network distinguishes between:
Real High-Resolution images
Generated Super-Resolved images

**🗺️ LULC Analysis**
The generated SR images are further processed using QGIS for:
Land Use and Land Cover classification
Urban expansion analysis
Environmental monitoring
Resource management

**📈 Results**
The SRGAN model successfully improves:
Spatial resolution
Image clarity
Texture details
LULC classification accuracy

The generated SR images are visually closer to HR images and provide better performance for remote sensing analysis.

**👩‍💻 Authors**
Snehitha Gudipudi(Corresponding Author)
Suryanarayana Gunnam
Priya Vyshnavi Jillellamudi
Dilleswara Sai Kiran Singuru
Department of ECE, Velagapudi Ramakrishna Siddhartha Engineering College, Vijayawada, India

**📜 License**
This project is developed for academic and research purposes.
