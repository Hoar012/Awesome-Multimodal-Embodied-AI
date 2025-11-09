# Multimodal-Embodied-AI

This repository collects papers, benchmarks, and datasets at the intersection of multimodal learning, embodied AI and robotics.

Continuously updating🔥🔥.
<!-- <p align="center">
    <img src="./images/MiG_logo.jpg" width="100%" height="100%">
</p>

## Our MLLM works

🔥🔥🔥 **A Survey on Multimodal Large Language Models**  
**[Project Page [This Page]](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)** | **[Paper](https://arxiv.org/pdf/2306.13549.pdf)** | :black_nib: **[Citation](./images/bib_survey.txt)** | **[💬 WeChat (MLLM微信交流群，欢迎加入)](./images/wechat-group.png)**

The first comprehensive survey for Multimodal Large Language Models (MLLMs). :sparkles:  

---

🔥🔥🔥 **VITA: Towards Open-Source Interactive Omni Multimodal LLM**  
<p align="center">
    <img src="./images/vita-1.5.jpg" width="60%" height="60%">
</p>

<font size=7><div align='center' > [[📽 VITA-1.5 Demo Show! Here We Go! 🔥](https://youtu.be/tyi6SVFT5mM?si=fkMQCrwa5fVnmEe7)] </div></font>  

<font size=7><div align='center' > [[📖 VITA-1.5 Paper](https://arxiv.org/pdf/2501.01957)] [[🌟 GitHub](https://github.com/VITA-MLLM/VITA)] [[🤖 Basic Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)] [[🍎 VITA-1.0](https://vita-home.github.io/)] [[💬 WeChat (微信)](https://github.com/VITA-MLLM/VITA/blob/main/asset/wechat-group.jpg)]</div></font>  

<font size=7><div align='center' > We are excited to introduce the **VITA-1.5**, a more powerful and more real-time version. ✨ </div></font>

<font size=7><div align='center' >**All codes of VITA-1.5 have been released**! :star2: </div></font>  

You can experience our [Basic Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo) on ModelScope directly. The Real-Time Interactive Demo needs to be configured according to the [instructions](https://github.com/VITA-MLLM/VITA?tab=readme-ov-file#-real-time-interactive-demo).


---

🔥🔥🔥 **Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuracy**  
<p align="center">
    <img src="./images/longvita.jpg" width="80%" height="80%">
</p>

<font size=7><div align='center' > [[📖 arXiv Paper](https://arxiv.org/pdf/2502.05177)] [[🌟 GitHub](https://github.com/VITA-MLLM/Long-VITA)]</div></font>  

<font size=7><div align='center' > Process more than **4K frames** or over **1M visual tokens**. State-of-the-art on Video-MME under 20B models!  ✨ </div></font>

--- -->

<font size=5><center><b> Table of Contents </b> </center></font>
- [Multimodal-Embodied-AI](#multimodal-embodied-ai)
- [Papers](#papers)
  - [Perception](#perception)
  - [Reasoning](#reasoning)
  - [Planning](#planning)
  - [Control](#control)
    - [Manipulation](#manipulation)
    - [Navigation](#navigation)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
  - [Perception](#perception-1)
  - [Reasoning](#reasoning-1)
  - [Planning](#planning-1)

# Papers

<!-- Template
|:--------|:--------:|:--------:|:--------:|
| [**Title**](Paperlink)  | Conference | [Page]( ) | [Github]( ) | -->

<!-- ## Foundation Models -->


## Perception

|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning**](https://arxiv.org/pdf/2505.12448?) | NeurIPS 2025 | [Page](https://yliu-cs.github.io/SSR/) | [Github](https://github.com/yliu-cs/SSR) |
| [**RaySt3R: Predicting Novel Depth Maps for Zero-Shot Object Completion**](https://arxiv.org/pdf/2506.05285?) | NeurIPS 2025 | [Page](https://rayst3r.github.io/) | [Github](https://github.com/Duisterhof/rayst3r) |
| [**AimBot: A Simple Auxiliary Visual Cue to Enhance Spatial Awareness of Visuomotor Policies**](https://arxiv.org/pdf/2508.08113) | NeurIPS 2025 | [Page](https://aimbot-reticle.github.io/) | [Github](https://github.com/aimbot-reticle/openpi0-aimbot) |
| [**RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics**](https://arxiv.org/pdf/2411.16537) | CVPR 2025 | [Page](https://chanh.ee/RoboSpatial/) | [Github](https://github.com/NVlabs/RoboSpatial) |
| [**Do vision-language models represent space and how? evaluating spatial frame of reference under ambiguities**](https://arxiv.org/pdf/2410.17385) | ICLR 2025 | [Page](https://spatial-comfort.github.io/) | [Github](https://github.com/sled-group/COMFORT) |
| [**Pre-training Auto-regressive Robotic Models with 4D Representations**](https://arxiv.org/pdf/2502.13142) | ICML 2025 | [Page](https://arm4r.github.io/) | [Github](https://github.com/Dantong88/arm4r) |
| [**Hearing the Slide: Acoustic-Guided Constraint Learning for Fast Non-Prehensile Transport**](https://arxiv.org/pdf/2506.09169) | CASE 2025 | [Page](https://fast-non-prehensile.github.io/) | - |
| [**Igniting VLMs toward the Embodied Space**](https://arxiv.org/pdf/2509.11766) | Arxiv | [Page](https://x2robot.com/en/research/68bc2cde8497d7f238dde690) | [Github](https://github.com/X-Square-Robot/wall-x) |
| [**RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics**](https://arxiv.org/pdf/2406.10721) | CoRL 2024 | [Page](https://robo-point.github.io/) | [Github](https://github.com/wentaoyuan/RoboPoint) |
| [**SonicBoom: Contact Localization Using Array of Microphones**](https://arxiv.org/pdf/2412.09878) | RAL 2024 | [Page](https://iamlab-cmu.github.io/sonicboom/) | [Github](https://github.com/markmlee/vibrotactile_localization) |
| [**RoboMP2: A Robotic Multimodal Perception-Planning Framework with Multimodal Large Language Models**](https://arxiv.org/pdf/2404.04929) | ICML 2024 | [Page](https://aopolin-lv.github.io/RoboMP2.github.io/) | [Github](https://github.com/aopolin-lv/RoboMP2) |
| [**OpenEQA: Embodied Question Answering in the Era of Foundation Models**](https://open-eqa.github.io/assets/pdfs/paper.pdf) | CVPR 2024 | [Page](https://open-eqa.github.io/) | [Github](https://github.com/facebookresearch/open-eqa) |
| [**EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI**](https://arxiv.org/pdf/2312.16170) | CVPR 2024 | [Page](https://tai-wang.github.io/embodiedscan/) | [Github](https://github.com/InternRobotics/EmbodiedScan/tree/main) |
| [**What’s Left? Concept Grounding with Logic-Enhanced Foundation Models**](https://arxiv.org/pdf/2310.16035.pdf) | NeurIPS 2023 | [Page](https://web.stanford.edu/~joycj/projects/left_neurips_2023.html) | [Github](https://github.com/joyhsu0504/LEFT/tree/main) |
| [**PACO: Parts and Attributes of Common Objects**](https://openaccess.thecvf.com/content/CVPR2023/papers/Ramanathan_PACO_Parts_and_Attributes_of_Common_Objects_CVPR_2023_paper.pdf) | CVPR 2023 | - | [Github](https://github.com/facebookresearch/paco) |
| [**Visuo-Acoustic Hand Pose and Contact Estimation**](https://arxiv.org/pdf/2508.00852) | Arxiv | - | - |
| [**Multimodal Perception for Goal-oriented Navigation: A Survey**](https://arxiv.org/pdf/2504.15643) | Arxiv | - | - |


## Reasoning

|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Gemini Robotics: Bringing AI into the Physical World**](https://arxiv.org/pdf/2503.20020) | Technical report | - | - |
| [**MolmoAct: Action Reasoning Models that can Reason in Space**](https://arxiv.org/pdf/2508.07917) | Technical report | [Page](https://allenai.org/blog/molmoact) | [Github](https://github.com/allenai/MolmoAct) |
| [**ChatVLA-2: Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge**](https://arxiv.org/pdf/2505.21906) | NeurIPS 2025 | [Page](https://chatvla-2.github.io/) | - |
| [**Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning**](https://arxiv.org/pdf/2503.15558) | Technical report | [Page](https://research.nvidia.com/labs/dir/cosmos-reason1/) | [Github](https://github.com/nvidia-cosmos/cosmos-reason1) |
| [**SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning**](https://arxiv.org/pdf/2504.20024) | NeurIPS 2025 | [Page](https://spatial-reasoner.github.io/) | [Github](https://github.com/johnson111788/SpatialReasoner) |
| [**RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics**](https://arxiv.org/pdf/2506.04308) | NeurIPS 2025 | [Page](https://zhoues.github.io/RoboRefer/) | [Github](https://github.com/Zhoues/RoboRefer) |
| [**Magma: A Foundation Model for Multimodal AI Agents**](https://www.arxiv.org/pdf/2502.13130) | CVPR 2025 | [Page](https://microsoft.github.io/Magma/) | [Github](https://github.com/microsoft/Magma) |
| [**RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete**](https://arxiv.org/pdf/2502.21257) | CVPR 2025 | [Page](https://superrobobrain.github.io/) | [Github](https://github.com/FlagOpen/RoboBrain) |
| [**RoboBrain 2.0 Technical Report**](https://arxiv.org/pdf/2507.02029) | Technical report | [Page](https://superrobobrain.github.io/) | [Github](https://github.com/FlagOpen/RoboBrain2.0) |
| [**Vlaser: Vision-Language-Action Model with Synergistic Embodied Reasoning**](https://arxiv.org/pdf/2510.11027) | Arxiv | [Page](https://internvl.github.io/blog/2025-10-11-Vlaser/) | [Github](https://github.com/OpenGVLab/Vlaser) |
| [**Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation**](https://arxiv.org/pdf/2508.13998) | Arxiv | [Page](https://embodied-r1.github.io/) | [Github](https://github.com/pickxiguapi/Embodied-R1) |
| [**PhysVLM: Enabling Visual Language Models to Understand Robotic Physical Reachability**](https://arxiv.org/pdf/2503.08481) | CVPR 2025 | - | [Github](https://github.com/unira-zwj/PhysVLM?tab=readme-ov-file) |
| [**ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation**](https://arxiv.org/pdf/2409.13682) | ICRA 2025 | [Page](https://nvidia-ai-iot.github.io/remembr/) | [Github](https://github.com/NVIDIA-AI-IOT/remembr) |
| [**InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning**](https://zifuwan.github.io/InstructPart/static/pdfs/ACL_2025_InstructPart.pdf) | ACL 2025 | [Page](https://zifuwan.github.io/InstructPart/) | [Github](https://github.com/zifuwan/InstructPart/tree/dataset) |
| [**SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models**](https://arxiv.org/pdf/2406.01584) | NeurIPS 2024 | [Page](https://www.anjiecheng.me/SpatialRGPT) | [Github](https://github.com/AnjieCheng/SpatialRGPT) |
| [**Multi-modal Situated Reasoning in 3D Scenes**](https://arxiv.org/pdf/2409.02389) | NeurIPS 2024 | [Page](https://msr3d.github.io/) | [Github](https://github.com/MSR3D/MSR3D) |
| [**EQA-MX: Embodied Question Answering using Multimodal Expression**](https://openreview.net/pdf?id=7gUrYE50Rb) | ICLR 2024 | - | [Github](https://github.com/mmiakashs/eqa-mx) |
| [**EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought**](https://openreview.net/pdf?id=7gUrYE50Rb) | NeurIPS 2023 | [Page](https://embodiedgpt.github.io/) | [Github](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) |
| [**Inner Monologue: Embodied Reasoning through Planning with Language Models**](https://arxiv.org/pdf/2207.05608) | CoRL 2022 | [Page](https://innermonologue.github.io/) | - |
| [**Robotic Control via Embodied Chain-of-Thought Reasoning**](https://arxiv.org/pdf/2407.08693) | Arxiv | [Page](https://embodied-cot.github.io/) | [Github](https://github.com/MichalZawalski/embodied-CoT) |
| [**Training Strategies for Efficient Embodied Reasoning**](https://arxiv.org/pdf/2505.08243) | Arxiv | [Page](https://ecot-lite.github.io/) | - |



## Planning

|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**EMBODIEDBENCH: Comprehensive Benchmarking Multi-modal Large  Language Models for Vision-Driven Embodied Agents**](https://arxiv.org/pdf/2502.09560) | ICML 2025 | [Page](https://embodiedbench.github.io/) | [Github](https://github.com/EmbodiedBench/EmbodiedBench) |


## Control

### Manipulation
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**π_0: A Vision-Language-Action Flow Model for General Robot Control**](https://arxiv.org/pdf/2410.24164v3) | RSS 2025 | [Page](https://www.physicalintelligence.company/blog/pi0) | [Github](https://github.com/Physical-Intelligence/openpi) |
| [**π_0.5:  a Vision-Language-Action Model with Open-World Generalization**](https://www.physicalintelligence.company/download/pi05.pdf) | CORL 2025 | [Page](https://www.physicalintelligence.company/blog/pi05) | [Github](https://github.com/Physical-Intelligence/openpi) |
| [**Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy**](https://robodita.github.io/dita.pdf) | ICCV 2025 | [Page](https://robodita.github.io/) | [Github](https://github.com/RoboDita/Dita) |
| [**MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting**](https://www.roboticsproceedings.org/rss20/p062.pdf) | RSS 2024 | [Page](https://moka-manipulation.github.io/) | [Github](https://github.com/moka-manipulation/moka) |
| [**Policy Blending and Recombination for Multimodal Contact-Rich Tasks**](https://www.ri.cmu.edu/app/uploads/2021/08/NaritaRAL2021.pdf) | RAL 2021 | - | - |



### Navigation
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**FLAME: Learning to Navigate with Multimodal LLM in Urban Environments**](https://arxiv.org/pdf/2408.11051) | AAAI 2025 | [Page](https://flame-sjtu.github.io/) | [Github](https://github.com/xyz9911/FLAME) |
| [**SmartWay: Enhanced Waypoint Prediction and Backtracking for Zero-Shot Vision-and-Language Navigation**](https://arxiv.org/pdf/2503.10069) | IROS 2025 | - | - |
| [**LLaDA: Driving Everywhere with Large Language Model Policy Adaptation**](https://arxiv.org/pdf/2402.05932) | CVPR 2024 | [Page](https://boyiliee.github.io/llada/) | [Github](https://github.com/Boyiliee/LLaDA-AV) |

<!-- ### Diffusion Policy -->


# Benchmarks and Datasets

## Perception
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**PhysBench Benchmarking and Enhancing VLMs for Physical World Understanding**](https://arxiv.org/pdf/2501.16411) | ICLR 2025 | [Page](https://physbench.github.io/) | [Github](https://github.com/USC-GVL/PhysBench) |
| [**OpenEQA: Embodied Question Answering in the Era of Foundation Models**](https://open-eqa.github.io/assets/pdfs/paper.pdf) | CVPR 2024 | [Page](https://open-eqa.github.io/) | [Github](https://github.com/facebookresearch/open-eqa) |
| [**Ost-bench: Evaluating The Capabilities Of Mllms In Online Spatio-temporal Scene Understanding**](https://arxiv.org/pdf/2507.07984) | Arxiv | [Page](https://rbler1234.github.io/OSTBench.github.io/) | [Github](https://github.com/OpenRobotLab/OST-Bench) |
| [**MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence**](https://arxiv.org/pdf/2505.23764) | Arxiv | [Page](https://runsenxu.com/projects/MMSI_Bench/) | [Github](https://github.com/InternRobotics/MMSI-Bench) |



## Reasoning
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Beyond the Destination: A Novel Benchmark for Exploration-Aware Embodied Question Answering**](https://arxiv.org/pdf/2503.11117) | ICCV 2025 | [Page](https://hcplab-sysu.github.io/EXPRESS-Bench/) | [Github](https://github.com/HCPLab-SYSU/EXPRESS-Bench) |
| [**OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models**](https://arxiv.org/pdf/2506.03135) | Arxiv | [Page](https://qizekun.github.io/omnispatial/) | [Github](https://github.com/qizekun/OmniSpatial) |





## Planning
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents**](https://arxiv.org/pdf/2502.09560) | ICML 2025 | [Page](https://embodiedbench.github.io/) | [Github](https://github.com/EmbodiedBench/EmbodiedBench) 
| [**EgoPlan-Bench: Benchmarking Multimodal Large Language Models for Human-Level Planning**](https://arxiv.org/pdf/2312.06722) | Arxiv | [Page](https://chenyi99.github.io/ego_plan/) | [Github](https://github.com/ChenYi99/EgoPlan) |
| [**WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning**](https://arxiv.org/pdf/2506.04363) | Arxiv | [Page](https://worldprediction.github.io/) | [Github](https://github.com/fairinternal/WorldPrediction) |




<!-- # Datasets -->

<!-- 
## Others
| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **IMAD** | [IMAD: IMage-Augmented multi-modal Dialogue](https://arxiv.org/pdf/2305.10512.pdf) | [Link](https://github.com/VityaVitalich/IMAD) | Multimodal dialogue dataset|
| **Video-ChatGPT** | [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://arxiv.org/pdf/2306.05424.pdf) | [Link](https://github.com/mbzuai-oryx/Video-ChatGPT#quantitative-evaluation-bar_chart) | A quantitative evaluation framework for video-based dialogue models | -->
