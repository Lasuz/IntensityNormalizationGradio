#____________________________________________________________
#            August 2, 2023
#____________________________________________________________
import os
import numpy as np
import gradio as gr
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt

def LoadImage(path_img):
    try:
        img_itk, voxel_size= Load_itk_image(path_img)
    except FileNotFoundError:
        print('ERROR: File not found', path_img)

    img_numpy = sitk.GetArrayFromImage(img_itk[0])
    img_load=img_numpy.transpose()

    #remove background = 0
    img_load[img_load==0]=np.nan

    return img_load


def Load_itk_image(path_img):
    #Read header
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(path_img)
    file_reader.ReadImageInformation()
    dim_total = file_reader.GetSize()
    img_vol_ITK = []

    if len(dim_total)==4:
        dim_vol = (dim_total[0:3])
        # make volume for all images
        img_load_all = sitk.ReadImage(path_img, sitk.sitkFloat32)

        img_vol_ITK = []
        for vol_n in range(dim_total[3]):
            #Extract vol out of object
            size = list(dim_total)
            size[3] = 0
            index = [0,0,0,vol_n]
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(size)
            extractor.SetIndex(index)

            img_vol_ITK.append(extractor.Execute(img_load_all))

    else:
        img_ITK = sitk.ReadImage(path_img, sitk.sitkFloat32)
        img_vol_ITK.append(img_ITK)
        dim_vol = dim_total

    #Voxelsize in mm. forth dimension has no meaning in spacing.
    voxel_size = img_vol_ITK[0].GetSpacing()
    return  img_vol_ITK, voxel_size



def NormalizeMinMax(img):
    #Normalization for Volumes
    minVal = np.nanmin(img)
    maxVal = np.nanmax(img)
 
    img_normalized = (img - minVal)/(maxVal- minVal)
    return img_normalized

def NormalizePercentile(img, minP, maxP):
    #Normalization for Volumes
    minVal = np.nanpercentile(img, minP)
    maxVal = np.nanpercentile(img, maxP)
    img_normalized = (img - minVal)/(maxVal- minVal)  
    return img_normalized

def NormalizeZScore(img):
    #Normalization for Volumes
    meanVal = np.nanmean(img)
    stdVal = np.nanstd(img)
     
    img_normalized = (img-meanVal)/stdVal
    
    return img_normalized
 

def PrepImage(pathName):
      
     
    img_load = LoadImage(pathName)
    img_normalized = NormalizeMinMax(img_load)
    img_normalized_per =NormalizePercentile(img_load, 10, 90)
    img_normalized_per98 =NormalizePercentile(img_load, 2, 98)
    img_normalized_zscore= (NormalizeZScore(img_load) ) 

    return img_load.flatten(),img_normalized.flatten(),img_normalized_per.flatten(),img_normalized_zscore.flatten(), img_normalized_per98.flatten()




def Norm_image(vol_path,normalization_technique):

    imglist = []
    imgClist=[]
    imgNlist=[]
    imgNPlist=[]
    imgNZlist=[]
    imgNPer98list=[]
    print(len(vol_path))
    for file in vol_path:
        pathName = file.name

        img_load = LoadImage(pathName)
        imglist.append(img_load.flatten())
        gr.Info('The volumes are succesfully loaded.')   

        if 'MinMax' in normalization_technique:
            imgNlist.append(NormalizeMinMax(img_load).flatten())
        
        if 'Z-Score' in normalization_technique:
            imgNZlist.append(NormalizeZScore(img_load).flatten())
        
        if 'Percentile (2th - 98th)'in normalization_technique:
            imgNPer98list.append(NormalizePercentile(img_load, 2, 98).flatten())
            
        if 'Percentile (10th - 90th)' in normalization_technique:
            imgNPlist.append(NormalizePercentile(img_load, 10, 90).flatten())

    gr.Info('The different normalization techniques are calculted.')   

    plt.figure(11)
    fig, ax = plt.subplots(1,1)
    for i, file in enumerate(vol_path):
        sns.histplot(data=imglist[i].flatten(), kde=False, label=os.path.basename(file.name)[0:25], log_scale=False,element="step", fill=False,bins=500,legend=True).set(title='Original')
    ax.legend()
    plt.savefig("Original.png")
    plots=["Original.png"]
    
    if 'MinMax' in normalization_technique:
        fig, ax = plt.subplots(1,1)
        for i, file in enumerate(vol_path):
            sns.histplot(data=imgNlist[i].flatten(), kde=False, label=os.path.basename(file.name)[0:25], log_scale=False,element="step", fill=False,bins=500,legend=True).set(title='Min-Max')
        ax.legend()
        plt.savefig("MinMax.png")
        plots.append("MinMax.png")
            
    if 'Z-Score' in normalization_technique:
        fig, ax = plt.subplots(1,1)
        for i, file in enumerate(vol_path):
            sns.histplot(data=imgNZlist[i].flatten(), kde=False, label=os.path.basename(file.name)[0:25], log_scale=False,element="step", fill=False,bins=500,legend=True).set(title='Z-Score')
        ax.legend()
        plt.savefig("Zscore.png")
        plots.append("Zscore.png")
        

    if 'Percentile (2th - 98th)'in normalization_technique:
        fig, ax = plt.subplots(1,1)
        for i, file in enumerate(vol_path):
            sns.histplot(data=imgNPer98list[i].flatten(), kde=False, label=os.path.basename(file.name)[0:25], log_scale=False,element="step", fill=False,bins=500,legend=True).set(title='Percentile 2-98')
        ax.legend()
        plt.savefig("Per98.png")
        plots.append("Per98.png")

    if 'Percentile (10th - 90th)' in normalization_technique:
        fig, ax = plt.subplots(1,1)
        for i, file in enumerate(vol_path):
            sns.histplot(data=imgNPlist[i].flatten(), kde=False, label=os.path.basename(file.name)[0:25], log_scale=False,element="step", fill=False,bins=500,legend=True).set(title='Percentile 10 - 90')
        ax.legend()
        plt.savefig("Per90.png")
        plots.append("Per90.png")
        
    gr.Info('The histograms are produced.')   

    return plots


description = 'You can upload mutiple image volumes (recommonded 3-5) in *.nii or *.nii.gz to see their histograms for multiple normalization techniques. Depending on file size and selected techniques it might take a while to do the calculations. \n The uploaded data is not stored and gets deleted once closing the window. '

inputs = [gr.File(file_count="multiple", label=None),gr.CheckboxGroup(["MinMax", "Z-Score", "Percentile (2th - 98th)", "Percentile (10th - 90th)"])]
demo = gr.Interface(fn=Norm_image,
             inputs=inputs,
             outputs=[gr.Gallery(label="Profiling Dashboard")], #.style(grid=(2,3))],
             description=description,
             )
             
demo.launch().queue()
