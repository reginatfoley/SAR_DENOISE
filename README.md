# sar_denoise
SAR images despeckling using deep learning neural networks.

Synthetic aperture radar (SAR) remote sensing offers a number of advantages over optical remote sensing. E.g. SAR image acquisition is not limited to daytime. Additionally, certain SAR radar friequencies can penetrate clouds, vegetation and can be used to provide important information about Earth surface and detect physical properties that are not possible with optical imagery.

However, the main drawback of SAR images is the presence of speckle, a signal dependent granular noise. It visually degrades the appearance of images and may severely affect analysis and information extraction from the images. For this reason SAR image speckle reduction, or despeckling, is of crucial importance for a number of applications. Most of the earlier despeckling techniques caused various degrees of blurring and loss of detail.

More recently, deep learning algorithms have been shown to achieve state-of the art performance in a variety of computer vision tasks, including image denoising. Sever studies have shown that deep learning can be successfully used in SAR image despeckling and it outperforms earlier despeckling techniques. In this project, I am planning to attempt SAR image despeckling with Stable Diffusion. To the best of my knowledge, such a study has not yet been applied to SAR image despeckling.