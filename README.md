# Benchmark-for-MEET
A Simple Benchmark for Few-Shot Learning in Remote Sensing Scene Classification on the MEET Dataset.

## A Brief Introduction to the MEET Dataset
The MEET (Million-scale finE-grained geospatial scEne classification dataseT) is a state-of-the-art, large-scale dataset designed for remote sensing scene classification. As one of the most comprehensive datasets in this domain, MEET comprises over 1.03 million high-resolution, unscaled remote sensing images spanning 80 fine-grained geospatial scene categories. A visual summary of the dataset is presented in Figure 1.

What distinguishes MEET from previous remote sensing scene classification datasets is its innovative "scene-in-scene" layout. Each image features a central scene (highlighted in red) that serves as the primary classification reference, while auxiliary scenes (outlined in green) provide valuable contextual information. This unique structure enhances the dataset’s ability to support fine-grained classification and contextual scene understanding.


![Fig1](https://github.com/user-attachments/assets/047f96e4-2ee3-4043-900d-8bc9850e4af3)
Fig1: MEET dataset overview

## Steps to Build MEET-FSL Dataset for Few-Shot Learning
Few-shot learning typically trains models on a very limited number of samples. So obviously, the MEET dataset can't be directly used for few-shot learning. To adapt it to the characteristics of few-shot learning, we limit the number of samples for each category by constructing subsets, with no more than 1,000 samples per category. We name this subset of MEET for few-shot learning as MEET-FSL. Here are the detailed steps for constructing the MEET-FSL dataset.

## Base and Novel Classes
When dividing the dataset into base and novel classes, we largely drew on the standards used in previous datasets, such as NWPU-RESISC45, WHU-RS19, UC-Merced, AID, and EuroSAT. Ultimately, we designated 40 categories as base classes and another 40 as novel classes. Of these novel classes, 15 were set aside for validation and 25 for testing, making MEET-FSL more challenging to some extent. The detailed class division is shown in Table 1.

| Dataset  | Dtrain                                                   | Dval                                                     | Dtest                                                    |
| :------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| MEET-FSL | Airport; Art center; Baseball field; Bridge; Buddhist temple; Chemical plant; Church; Christmas tree farm; Cloud; Coffee plantation; Construction; Dam; Dry field; Driving school; Football field; Glacier; Highway; Lake; Landfill; Low-rise residential area; Military center; Mosque; Mountain; Office; Overpass; Paddy field; Port; Railway; Resort; Rock; Roundabout; Scrub; Solar power plant; Stadium; Substation; Theme park; Tennis court; Urban village; Vacant parking lot; Wetland. | American football field; Beach; Busy parking lot; Cemetery; Gas station; Hospital; Mangrove; Orchard; Prison; Quarry; Retail; Swimming pool; Tollbooth; Water plants; Wind power plant. | Basketball court; Brownfield; College; Commercial; Farmyard; Forest; Golf; Greenhouse; High-rise residential area; Highway service area; Hills; Meadow; Military training ground; Park; Pond; Petroleum well; Railway station; River; Sand; School; Square; Steel plant; Tea plantation; Thermal power plant; Village. |

# Random Sampling
For each sample category, we performed a triple-split, ensuring each split had no more than 1,000 samples. This created three subsets, namely Split1, Split2, and Split3, for the MEET-FSL dataset. The size of MEET-FSL is 100GB, roughly one-sixth of the original MEET dataset. To ensure reproducibility, we set the random splitting seed to 42.

# Implementation Details of the DBA-RMCL Method on the MEET-FSL Dataset
Our DBA-RMCL approach does not specially handle the "scene-in-scene" characteristic of the MEET dataset. Instead, we directly sample the central scenes to establish DBA-RMCL's baseline results on MEET-FSL. For detailed data processing on this dataset, please refer to Algorithm 2.

```python
# Pseudocode of Data Preprocessing, PyTorch-like
image_size = 256  # Unify image size
augmentation = [  # Train Only
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop(84),  # Use the center scene
    transforms.RandomResizedCrop(70, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize()
]
```

## Visual Analysis
In the MEET-split1 dataset, we utilized t-SNE to visualize the class feature distributions of DBA-RMCL and ProtoNet (as shown in Figures 2 and 3). It is evident that traditional metric learning methods like ProtoNet exhibit limited clustering performance, forming only loosely defined clusters with insufficient intra-class compactness and weak inter-class separability. This phenomenon is primarily due to the increased task complexity, which leads to more diverse sample representations and, consequently, greater classification challenges.

In contrast, our proposed DBA-RMCL demonstrates significantly stronger feature aggregation capabilities in this scenario. Specifically, it not only facilitates the formation of well-defined class clusters but also enhances intra-class compactness while establishing clearer separations between different classes. However, DBA-RMCL still has certain limitations. For instance, in the left region of the t-SNE visualization, some class centroids remain relatively close to each other, leading to a degree of feature overlap between categories, which constrains the model’s discriminative ability.

Overall, while DBA-RMCL achieves significant performance improvements in few-shot classification tasks for remote sensing imagery, there remains room for further optimization, especially when tackling more complex challenges. Future work could focus on enhancing the model’s adaptability to different way and shot configurations, improving its generalization under diverse task conditions, and increasing its robustness to variations in task difficulty.

![Fig14](https://github.com/user-attachments/assets/4f0cbd20-3e32-4a3a-aab7-7179c62955de)
Fig2: t-SNE Visualization Results on the MEET-split1 Dataset using ProtoNet
![Fig15](https://github.com/user-attachments/assets/f561df26-a838-48bd-980a-50d0f0ea63f0)
Fig3: t-SNE Visualization Results on the MEET-split1 Dataset using DBA-RMCL
