## Project Overview
Team members: Xiaoran Du, Simon Liu (Department of Computer Science, University of Illinois at Urbana-Champaign)
### Motivation
[This project](https://github.com/xiaorandu/pix2pix_zero/blob/main/cs445.ipynb) is focusing on implementing the [pix2pix-zero paper](https://arxiv.org/pdf/2302.03027)[1] due to its novel approach to
zero-shot image-to-image translation, leveraging pre-trained text-to-image diffusion models
without the need for task-specific training or manual text prompting. Our interest lies in
understanding and assessing the effectiveness of its unique mechanism for automatic editing
direction discovery and cross-attention guidance for content preservation. Through this project,
we hope to delve deep into the mechanics of pix2pix-zero, validating its claimed capacities and
exploring its potential limitations. We also want to expand our understanding of diffusion models
and their practical applications in real-world image editing.

### Approach
The project implements the core pipeline method in the pix2pix-zero approach, including
inverting real images, discovering edit directions, and cross-attention guidance.

In the pipeline, text captions are generated from an input image using BLIP, followed by the
creation of an inverse noise map through regularized DDIM inversion. Reference cross-attention
maps are then produced to match the structure of the image, guided by the CLIP embeddings of
the generated text. The process continues with denoising, using the edited text embeddings,
focusing on aligning the current cross-attention maps with the reference ones.

While most of the results of this project use the cat2dog task, which is available in the pix2pix-
zero demo code, new edit directions were created by generating large amounts of image prompts
related to a keyword, then computing an average text embedding for that keyword. These pre-
computed embeddings allow users to define edit directions on the fly – the pipeline achieves this
by taking the difference between two embeddings.

We perform quantitative evaluations using three metrics: CLIP Acc, Structure Dist, and LPIPS
error, comparing pix2pix-zero approach to some previous and concurrent diffusion-based image
editing methods.

One of the limitations of pix2pix-zero is that the edits do not perform well on images with
subjects that are not oriented properly and are rotated a certain amount. Our project also explores
attempts to mitigate this limitation by applying transformations to the input image before and
after the edit to produce better results.

### Implementation details
For image inversion, we implement the deterministic DDIM reverse process proposed by Song,
Jiaming et al. [2] for converting real images into their latent representations. The inversion
process utilizes a scheduler to control noise levels and applies the denoising model (U-Net)
conditioned on text embeddings. Classifier-free guidance is integrated during this process. The
generated inversed noise map is then regulated by auto-correlation and KL divergence
regularization techniques to ensure the noise predictions remain well-behaved, aiding in
preserving image quality and structural integrity during inversion.

Text embeddings are produced by first generating sentences with Google’s Flan-T5 model, then
encoding using the CLIP ViT-L/14 encoder which is used in the CompVis/Stable-Diffusion-v1-4
model. After getting an embedding for each sentence, all embeddings are averaged and saved to
a file for later use. Edit directions are created by computing the difference between two
embedding files.

Editing via Cross-Attention Guidance is performed as described in the original paper, by first
reconstructing the image without performing the edit directions, and then generating the cross
attention maps using the edit directions. Finally, we take a gradient step toward the matching
reference.

Automatic image rotation is implemented by taking an input image and rotating it in 30-degree
increments between 0 and 360 degrees before running it through the edit pipeline and rotating it
back. Prior to rotating it back, a score of the target object is generated using Facebook
Research’s Detic model and associated to the edited image. The image with the highest
confidence score is then recommended as the best result.

For quantitative evaluation metrics, we utilize CLIP score introduced by Hessel, Jack, et al. [3] to
assess whether the edit aligns successfully with the target attributes, with a higher score refers
indicating greater similarity between the edited image and the target text. We also implement the
method proposed by Tumanyan, Narek, et al. [4] for measuring structural distance between two
images using a pre-trained Vision Transformer (ViT) model DINO-Vit [6]. This metric leverages
deep spatial features extracted from DINO-Vit, using their self-similarity as a structure
representation. A lower Structure Dist score indicates that the structure of the edited image
closely resembles that of the input image. The LPIP metric, proposed by Zhang, Richard, et al.
[5], measures perceptual similarity between images utilizing deep learning features from AlexNet
and VGG. A lower LPIP error score means more similarity between the images. In our project,
we measure the LPIP errors using both AlexNet and VGG models.

### Results
We have successfully implemented the pipeline method proposed in the pix2pix-zero paper and
obtained the expected results. We compared the pix2pix-zero approach with baseline models
including SDEdit + word swap, Prompt-to-Prompt, and DDIM + word swap, applying the
quantitative evaluation metrics. The metrics demonstrate that pix2pix-zero achieves a high CLIP
score while maintaining low Structure Dist and LPIPS error, indicating effective edit
performance while preserving the structure of the image.

### Reference
1. Parmar, Gaurav, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-
Yan Zhu. "Zero-shot image-to-image translation." In ACM SIGGRAPH 2023
Conference Proceedings, pp. 1-11. 2023.
2. Jiaming Song, Chenlin Meng, and Stefano Ermon. “Denoising diffusion implicit
models”. In International Conference on Learning Representations (ICLR), 2021.
3. Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan LeBras, and Yejin Choi.
“CLIPScore: a reference-free evaluation metric for image captioning”. In EMNLP, 2021.
4. Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. “Splicing vit features for
semantic appearance transfer”. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp.10748–10757, 2022.
5. Zhang, Richard, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. "The
unreasonable effectiveness of deep features as a perceptual metric." In Proceedings of the
IEEE conference on computer vision and pattern recognition, pp. 586-595. 2018.
6. Self-Supervised Vision Transformers with DINO:
https://github.com/facebookresearch/dino
