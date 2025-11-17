# ⚙️ Multimodal-Embodied-AI

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
@[toc]
## 📋 Table of Contents
- [⚙️ Multimodal-Embodied-AI](#multimodal-embodied-ai)
- [📄 Papers](#-papers)
  - [Perception](#perception)
  - [Reasoning](#reasoning)
  - [Planning](#planning)
  - [Control](#control)
    - [Manipulation](#manipulation)
    - [Navigation](#navigation)
- [📊 Benchmarks & Datasets](#-benchmarks-&-datasets)
  - [Perception](#perception-1)
  - [Reasoning](#reasoning-1)
  - [Planning](#planning-1)
  - [Control](#control-1)
    - [Manipulation](#manipulation-1)
    - [Navigation](#navigation-1)
    
## 📄 Papers

<!-- Template
|:--------|:--------:|:--------:|:--------:|
| [**Title**](Paperlink) | Conference | [Page](link) | [Github](link) |
 -->

<!-- ## Foundation Models -->


### Perception
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning**](https://arxiv.org/pdf/2505.12448?) | NeurIPS 2025 | [Page](https://yliu-cs.github.io/SSR/) | [Github](https://github.com/yliu-cs/SSR) |
| [**From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D**](https://arxiv.org/pdf/2503.22976) | NeurIPS 2025 | [Page](https://logosroboticsgroup.github.io/SPAR/) | [Github](https://github.com/LogosRoboticsGroup/SPAR) |
| [**RaySt3R: Predicting Novel Depth Maps for Zero-Shot Object Completion**](https://arxiv.org/pdf/2506.05285?) | NeurIPS 2025 | [Page](https://rayst3r.github.io/) | [Github](https://github.com/Duisterhof/rayst3r) |
| [**AimBot: A Simple Auxiliary Visual Cue to Enhance Spatial Awareness of Visuomotor Policies**](https://arxiv.org/pdf/2508.08113) | NeurIPS 2025 | [Page](https://aimbot-reticle.github.io/) | [Github](https://github.com/aimbot-reticle/openpi0-aimbot) |
| [**EmbodiedSAM: Online Segment Any 3D Thing in Real Time**](https://arxiv.org/pdf/2408.11811) | ICLR 2025 | [Page](https://xuxw98.github.io/ESAM/) | [Github](https://github.com/xuxw98/ESAM) |
| [**GrabS: Generative Embodied Agent for 3D Object Segmentation without Scene Supervision**](https://arxiv.org/pdf/2504.11754) | ICLR 2025 | - | [Github](https://github.com/vLAR-group/GrabS) |
| [**RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics**](https://arxiv.org/pdf/2411.16537) | CVPR 2025 | [Page](https://chanh.ee/RoboSpatial/) | [Github](https://github.com/NVlabs/RoboSpatial) |
| [**Do vision-language models represent space and how? evaluating spatial frame of reference under ambiguities**](https://arxiv.org/pdf/2410.17385) | ICLR 2025 | [Page](https://spatial-comfort.github.io/) | [Github](https://github.com/sled-group/COMFORT) |
| [**Pre-training Auto-regressive Robotic Models with 4D Representations**](https://arxiv.org/pdf/2502.13142) | ICML 2025 | [Page](https://arm4r.github.io/) | [Github](https://github.com/Dantong88/arm4r) |
| [**Hearing the Slide: Acoustic-Guided Constraint Learning for Fast Non-Prehensile Transport**](https://arxiv.org/pdf/2506.09169) | CASE 2025 | [Page](https://fast-non-prehensile.github.io/) | - |
| [**PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding**](https://arxiv.org/pdf/2504.13180)  | Arxiv | [Page](https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding/) | [Github](https://github.com/facebookresearch/perception_models) |
| [**Igniting VLMs toward the Embodied Space**](https://arxiv.org/pdf/2509.11766) | Arxiv | [Page](https://x2robot.com/en/research/68bc2cde8497d7f238dde690) | [Github](https://github.com/X-Square-Robot/wall-x) |
| [**RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics**](https://arxiv.org/pdf/2406.10721) | CoRL 2024 | [Page](https://robo-point.github.io/) | [Github](https://github.com/wentaoyuan/RoboPoint) |
| [**SonicBoom: Contact Localization Using Array of Microphones**](https://arxiv.org/pdf/2412.09878) | RAL 2024 | [Page](https://iamlab-cmu.github.io/sonicboom/) | [Github](https://github.com/markmlee/vibrotactile_localization) |
| [**Embodied Uncertainty-Aware Object Segmentation**](https://arxiv.org/pdf/2408.04760) | IROS 2024 | [Page](https://sites.google.com/view/embodied-uncertain-seg) | [Github](https://github.com/FANG-Xiaolin/uncos) |
| [**RoboMP2: A Robotic Multimodal Perception-Planning Framework with Multimodal Large Language Models**](https://arxiv.org/pdf/2404.04929) | ICML 2024 | [Page](https://aopolin-lv.github.io/RoboMP2.github.io/) | [Github](https://github.com/aopolin-lv/RoboMP2) |
| [**OpenEQA: Embodied Question Answering in the Era of Foundation Models**](https://open-eqa.github.io/assets/pdfs/paper.pdf) | CVPR 2024 | [Page](https://open-eqa.github.io/) | [Github](https://github.com/facebookresearch/open-eqa) |
| [**EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI**](https://arxiv.org/pdf/2312.16170) | CVPR 2024 | [Page](https://tai-wang.github.io/embodiedscan/) | [Github](https://github.com/InternRobotics/EmbodiedScan/tree/main) |
| [**What’s Left? Concept Grounding with Logic-Enhanced Foundation Models**](https://arxiv.org/pdf/2310.16035.pdf) | NeurIPS 2023 | [Page](https://web.stanford.edu/~joycj/projects/left_neurips_2023.html) | [Github](https://github.com/joyhsu0504/LEFT/tree/main) |
| [**PACO: Parts and Attributes of Common Objects**](https://openaccess.thecvf.com/content/CVPR2023/papers/Ramanathan_PACO_Parts_and_Attributes_of_Common_Objects_CVPR_2023_paper.pdf) | CVPR 2023 | - | [Github](https://github.com/facebookresearch/paco) |
| [**Interactron: Embodied Adaptive Object Detection**](https://arxiv.org/pdf/2202.00660) | CVPR 2022 | - | [Github](https://github.com/allenai/interactron) |
| [**Visuo-Acoustic Hand Pose and Contact Estimation**](https://arxiv.org/pdf/2508.00852) | Arxiv | - | - |
| [**Multimodal Perception for Goal-oriented Navigation: A Survey**](https://arxiv.org/pdf/2504.15643) | Arxiv | - | - |


### Reasoning

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
| [**RoboVQA: Multimodal Long-Horizon Reasoning for Robotics**](https://arxiv.org/pdf/2311.00899) | ICRA 2024 | [Page](https://robovqa.github.io/) | [Github](https://github.com/google-deepmind/robovqa) |
| [**SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models**](https://arxiv.org/pdf/2406.01584) | NeurIPS 2024 | [Page](https://www.anjiecheng.me/SpatialRGPT) | [Github](https://github.com/AnjieCheng/SpatialRGPT) |
| [**Multi-modal Situated Reasoning in 3D Scenes**](https://arxiv.org/pdf/2409.02389) | NeurIPS 2024 | [Page](https://msr3d.github.io/) | [Github](https://github.com/MSR3D/MSR3D) |
| [**EQA-MX: Embodied Question Answering using Multimodal Expression**](https://openreview.net/pdf?id=7gUrYE50Rb) | ICLR 2024 | - | [Github](https://github.com/mmiakashs/eqa-mx) |
| [**EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought**](https://openreview.net/pdf?id=7gUrYE50Rb) | NeurIPS 2023 | [Page](https://embodiedgpt.github.io/) | [Github](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) |
| [**Inner Monologue: Embodied Reasoning through Planning with Language Models**](https://arxiv.org/pdf/2207.05608) | CoRL 2022 | [Page](https://innermonologue.github.io/) | - |
| [**Robotic Control via Embodied Chain-of-Thought Reasoning**](https://arxiv.org/pdf/2407.08693) | Arxiv | [Page](https://embodied-cot.github.io/) | [Github](https://github.com/MichalZawalski/embodied-CoT) |
| [**Training Strategies for Efficient Embodied Reasoning**](https://arxiv.org/pdf/2505.08243) | Arxiv | [Page](https://ecot-lite.github.io/) | - |


### Planning

|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**EMBODIEDBENCH: Comprehensive Benchmarking Multi-modal Large  Language Models for Vision-Driven Embodied Agents**](https://arxiv.org/pdf/2502.09560) | ICML 2025 | [Page](https://embodiedbench.github.io/) | [Github](https://github.com/EmbodiedBench/EmbodiedBench) |
| [**Embodied large language models enable robots to complete complex tasks in unpredictable environments**](https://www.nature.com/articles/s42256-025-01005-x.pdf) | Nature Machine Intelligence 2025 | - | - |
| [**DELTA: Decomposed Efficient Long-Term Robot Task Planning using Large Language Models**](https://arxiv.org/pdf/2404.03275) | ICRA 2025 | [Page](https://delta-llm.github.io/) | [Github](https://github.com/boschresearch/DELTA) |
| [**MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents**](https://arxiv.org/pdf/2410.03450) | ICLR 2025 | - | - |
| [**Multimodal LLM Guided Exploration and Active Mapping using Fisher Information**](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Multimodal_LLM_Guided_Exploration_and_Active_Mapping_using_Fisher_Information_ICCV_2025_paper.pdf) | ICCV 2025 | - | - |
| [**Open-World Planning via Lifted Regression with LLM-Inferred Affordances for Embodied Agents**](https://aclanthology.org/2025.acl-long.1018.pdf) | ACL 2025 | - | - |
| [**Structured Preference Optimization for Vision-Language Long-Horizon Task Planning**](https://aclanthology.org/2025.emnlp-main.884.pdf) | EMNLP 2025 | - | - |
| [**Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation**](https://arxiv.org/pdf/2506.22827) | RSS Workshop 2025 | [Page](https://vlp-humanoid.github.io/) | - |
| [**Multi-Modal Grounded Planning and Efficient Replanning For Learning Embodied Agents with A Few Examples**](https://arxiv.org/pdf/2412.17288) | AAAI 2025 | [Page](https://twoongg.github.io/projects/flare/) | [Github](https://github.com/snumprlab/flare) |
| [**Safe planner: Empowering safety awareness in large pre-trained models for robot task planning**](https://arxiv.org/pdf/2411.06920) | AAAI 2025 | [Page](https://sites.google.com/view/safeplanner) | - |
| [**Pre-emptive Action Revision by Environmental Feedback for Embodied Instruction Following Agents**](https://openreview.net/pdf?id=cq2uB30uBM) | CoRL 2024 | [Page](https://pred-agent.github.io/) | [Github](https://github.com/snumprlab/pred) |
| [**LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning**](https://ll3da.github.io/static/files/LL3DA__Visual_Interactive_Instruction_Tuning_for_Omni_3D_Understanding__Reasoning__and_Planning.pdf) | CVPR 2024 | [Page](https://ll3da.github.io/) | [Github](https://github.com/Open3DA/LL3DA) |
| [**RILA: Reflective and Imaginative Language Agent for Zero-Shot Semantic Audio-Visual Navigation**](https://peihaochen.github.io/files/publications/RILA.pdf) | CVPR 2024 | - | - |
| [**Multimodal Procedural Planning via Dual Text-Image Prompting**](https://aclanthology.org/2024.findings-emnlp.641.pdf) | EMNLP 2024 | - | - |
| [**Embodied Agent Interface: A Single Line to Evaluate LLMs for Embodied Decision Making**](https://arxiv.org/pdf/2410.07166) | NeurIPS 2024 | [Page](https://embodied-agent-interface.github.io/) | [Github](https://github.com/embodied-agent-interface/embodied-agent-interface) |
| [**Exploratory Retrieval-Augmented Planning For Continual Embodied Instruction Following**](https://arxiv.org/pdf/2509.08222) | NeurIPS 2024 | - | - |
| [**Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld**](https://arxiv.org/pdf/2311.16714) | CVPR 2024 | - | [Github](https://github.com/stevenyangyj/Emma-Alfworld) |
| [**Plan-seq-learn: Language model guided rl for solving long horizon robotics tasks**](https://arxiv.org/pdf/2405.01534) | ICLR 2024 | [Page](https://mihdalal.github.io/planseqlearn/) | [Github](https://github.com/mihdalal/planseqlearn) |
| [**SayNav: Grounding Large Language Models for Dynamic Planning to Navigation in New Environments**](https://ojs.aaai.org/index.php/ICAPS/article/download/31506/33666) | AAAI 2024 | [Page](https://www.sri.com/publication/saynav-grounding-large-language-models-for-dynamic-planning-to-navigation-in-new-environments/) | [Github](https://github.com/arajv/SayNav) |
| [**Learning Adaptive Planning Representations with Natural Language Guidance**](https://arxiv.org/pdf/2312.08566.pdf) | ICLR 2024 | [Page](https://concepts-ai.com/p/ada/) | [Github](https://github.com/CatherineWong/llm-operators/) |
| [**What Planning Problem Can A Relational Neural Network Solve**](https://arxiv.org/pdf/2312.03682.pdf) | NeurIPS 2023 | [Page](https://concepts-ai.com/p/goal-regression-width/) | [Github](https://github.com/concepts-ai/goal-regression-width) |
| [**Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents**](https://arxiv.org/pdf/2302.01560) | NeurIPS 2023 | [Page](http://www.craftjarvis.org/) | [Github](https://github.com/CraftJarvis/MC-Planner) |
| [**Sayplan: Grounding large language models using 3d scene graphs for scalable robot task planning**](https://arxiv.org/pdf/2307.06135) | CoRL 2023 | [Page](https://sayplan.github.io/) | - |
| [**Grounded Decoding: Guiding Text Generation with Grounded Models for Robot Control**](https://grounded-decoding.github.io/paper.pdf) | NeurIPS 2023 | [Page](https://grounded-decoding.github.io/) | - |
| [**Do Embodied Agents Dream of Pixelated Sheep? Embodied Decision Making using Language Guided World Modelling**](https://arxiv.org/pdf/2301.12050) | ICML 2023 | [Page](https://deckardagent.github.io/) | [Github](https://github.com/DeckardAgent/deckard) |
| [**Learning Neuro-Symbolic Skills for Bilevel Planning**](https://arxiv.org/pdf/2206.10680) | CoRL 2022 | - | [Github](https://github.com/Learning-and-Intelligent-Systems/predicators) |
| [**Learning Neuro-Symbolic Relational Transition Models for Bilevel Planning**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9981440) | IROS 2022 | - | [Github](https://github.com/Learning-and-Intelligent-Systems/predicators) |
| [**LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models**](https://arxiv.org/pdf/2212.04088) | ICCV 2023 | [Page](https://dki-lab.github.io/LLM-Planner/) | [Github](https://github.com/OSU-NLP-Group/LLM-Planner/) |
| [**Context-Aware Planning and Environment-Aware Memory for Instruction Following Embodied Agents**](https://arxiv.org/pdf/2308.07241v2) | ICCV 2023 | [Page](https://bhkim94.github.io/projects/CAPEAM/) | [Github](https://github.com/snumprlab/capeam) |
| [**Omnieva: Embodied versatile planner via task-adaptive 3d-grounded and embodiment-aware reasoning**](https://arxiv.org/pdf/2509.09332) | Arxiv | [Page](https://omnieva.github.io/) | - |
| [**Preference-Based Long-Horizon Robotic Stacking with Multimodal Large Language Models**](https://arxiv.org/pdf/2509.24163) | Arxiv | - | - |
| [**Reinforced Embodied Planning with Verifiable Reward for Real-World Robotic Manipulation**](https://arxiv.org/pdf/2509.25852) | Arxiv | - | - |
| [**Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning**](https://robot-vila.github.io/ViLa.pdf) | Arxiv | [Page](https://robot-vila.github.io/) | - |
| [**Embodied Task Planning with Large Language Models**](https://arxiv.org/pdf/2307.01848) | Arxiv | [Page](https://gary3410.github.io/TaPA/) | [Github](https://github.com/Gary3410/TaPA) |








### Control

#### Manipulation
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success**](https://arxiv.org/pdf/2502.19645) | RSS 2026 | [Page](https://openvla-oft.github.io/) | [Github](https://github.com/moojink/openvla-oft) |
| [**Grounding Actions in Camera Space: Observation-Centric Vision-Language-Action Policy**](https://arxiv.org/pdf/2508.13103) | AAAI 2026 | - | - |
| [**ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning**](https://arxiv.org/pdf/2507.16815) | NeurIPS 2025 | [Page](https://jasper0314-huang.github.io/thinkact-vla/) | - |
| [**BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models**](https://arxiv.org/pdf/2506.07961) | NeurIPS 2025 | [Page](https://bridgevla.github.io/) | [Github](https://github.com/BridgeVLA/BridgeVLA) |
| [**What Can RL Bring to VLA Generalization? An Empirical Study**](https://arxiv.org/pdf/2505.19789) | NeurIPS 2025 | [Page](https://rlvla.github.io/) | [Github](https://github.com/gen-robot/RL4VLA) |
| [**DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control**](https://arxiv.org/pdf/2502.05855) | CoRL 2025 | [Page](https://dex-vla.github.io/) | [Github](https://github.com/bytenaija/dexvla) |
| [**GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data**](https://arxiv.org/pdf/2505.03233) | CoRL 2025 | [Page](https://pku-epic.github.io/GraspVLA-web/) | [Github](https://github.com/PKU-EPIC/GraspVLA) |
| [**RoboBERT: An End-to-end Multimodal Robotic Manipulation Model**](https://arxiv.org/pdf/2502.07837) | CoRL 2025 | [Page](https://anonymeskonto.github.io/Web/) | [Github](https://github.com/AnonymesKonto/RoboBERT) |
| [**Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation**](https://long-vla.github.io/long-vla/longvla.pdf) | CoRL 2025 | [Page](https://long-vla.github.io/) | - |
| [**Learning to Act Anywhere with Task-centric Latent Actions**](https://arxiv.org/pdf/2505.06111) | RSS 2025 | - | [Github](https://github.com/OpenDriveLab/UniVLA) |
| [**ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy**](https://arxiv.org/pdf/2502.05450) | RSS 2025 | [Page](https://cccedric.github.io/conrft/) | [Github](https://github.com/cccedric/conrft) |
| [**Gemini Robotics: Bringing AI into the Physical World**](https://arxiv.org/pdf/2503.20020) | Technical report | - | - |
| [**π_0: A Vision-Language-Action Flow Model for General Robot Control**](https://arxiv.org/pdf/2410.24164v3) | RSS 2025 | [Page](https://www.physicalintelligence.company/blog/pi0) | [Github](https://github.com/Physical-Intelligence/openpi) |
| [**π_0.5:  a Vision-Language-Action Model with Open-World Generalization**](https://www.physicalintelligence.company/download/pi05.pdf) | CORL 2025 | [Page](https://www.physicalintelligence.company/blog/pi05) | [Github](https://github.com/Physical-Intelligence/openpi) |
| [**Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy**](https://robodita.github.io/dita.pdf) | ICCV 2025 | [Page](https://robodita.github.io/) | [Github](https://github.com/RoboDita/Dita) |
| [**RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation**](https://arxiv.org/pdf/2410.07864) | ICLR 2025 | [Page](https://rdt-robotics.github.io/rdt-robotics/) | [Github](https://github.com/thu-ml/RoboticsDiffusionTransformer) |
| [**CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models**](https://arxiv.org/pdf/2503.22020) | CVPR 2025 | [Page](https://cot-vla.github.io/) | - |
| [**RoboGround: Robotic Manipulation with Grounded Vision-Language Priors**](https://arxiv.org/pdf/2504.21530) | CVPR 2025 | [Page](https://robo-ground.github.io/) | [Github](https://github.com/ZzZZCHS/RoboGround) |
| [**Magma: A Foundation Model for Multimodal AI Agents**](https://www.arxiv.org/pdf/2502.13130) | CVPR 2025 | [Page](https://microsoft.github.io/Magma/) | [Github](https://github.com/microsoft/Magma) |
| [**OpenVLA: An Open-Source Vision-Language-Action Model**](https://arxiv.org/pdf/2406.09246) | CoRL 2024 | [Page](https://openvla.github.io/) | [Github](https://github.com/openvla/openvla) |
| [**Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation**](https://arxiv.org/pdf/2312.13139) | ICLR 2024 | [Page](https://gr1-manipulation.github.io/) | [Github](https://github.com/bytedance/GR-1) |
| [**GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation**](https://arxiv.org/pdf/2410.06158) | Technical report | [Page](https://gr2-manipulation.github.io/) | - |
| [**GR-3 Technical Report**](https://arxiv.org/pdf/2507.15493) | Technical report | [Page](https://seed.bytedance.com/en/GR3) | - |
| [**MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting**](https://www.roboticsproceedings.org/rss20/p062.pdf) | RSS 2024 | [Page](https://moka-manipulation.github.io/) | [Github](https://github.com/moka-manipulation/moka) |
| [**RT-1: Robotics Transformer for Real-World Control at Scale**](https://arxiv.org/pdf/2212.06817) | RSS 2023 | [Page](https://robotics-transformer1.github.io/) | [Github](https://github.com/google-research/robotics_transformer) |
| [**RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**](https://robotics-transformer2.github.io/assets/rt2.pdf) | Technical report | [Page](https://robotics-transformer2.github.io/) | [Github](https://github.com/google-research/robotics_transformer) |
| [**Octo: An Open-Source Generalist Robot Policy**](https://arxiv.org/pdf/2405.12213) | RSS 2024 | [Page](https://octo-models.github.io/) | [Github](https://github.com/octo-models/octo) |
| [**Open X-Embodiment: Robotic Learning Datasets and RT-X Models**](https://arxiv.org/pdf/2310.08864) | ICRA 2024 | [Page](https://robotics-transformer-x.github.io/) | [Github](https://github.com/google-deepmind/open_x_embodiment) |
| [**VIMA: General Robot Manipulation with Multimodal Prompts**](https://vimalabs.github.io/assets/vima_paper.pdf) | ICML 2023 | [Page](https://vimalabs.github.io/) | [Github](https://github.com/vimalabs/VIMA) |
| [**Policy Blending and Recombination for Multimodal Contact-Rich Tasks**](https://www.ri.cmu.edu/app/uploads/2021/08/NaritaRAL2021.pdf) | RAL 2021 | - | - |




#### Navigation
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**NaVILA: Legged Robot Vision-Language-Action Model for Navigation**](https://navila-bot.github.io/static/navila_paper.pdf) | RSS 2025 | [Page](https://navila-bot.github.io/) | [Github](https://github.com/AnjieCheng/NaVILA) |
| [**Multimodal Spatial Language Maps for Robot Navigation and Manipulation**](https://arxiv.org/pdf/2506.06862) | IJRR 2025 | [Page](https://mslmaps.github.io/) | [Github](https://github.com/vlmaps/VLMaps) |
| [**ApexNav: An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion**](https://arxiv.org/pdf/2504.14478) | RA-L 2025 | [Page](https://robotics-star.com/ApexNav/) | [Github](https://github.com/Robotics-STAR-Lab/ApexNav) |
| [**Search-TTA: A Multimodal Test-Time Adaptation Framework for Visual Search in the Wild**](https://arxiv.org/pdf/2505.11350) | CoRL 2025 | [Page](https://search-tta.github.io/) | [Github](https://github.com/marmotlab/Search-TTA-RL-VLN) |
| [**GC-VLN: Instruction as Graph Constraints for Training-free Vision-and-Language Navigation**](https://arxiv.org/pdf/2509.10454) | CoRL 2025 | [Page](https://bagh2178.github.io/GC-VLN/) | [Github](https://github.com/bagh2178/GC-VLN) |
| [**RoboTron-Nav: A Unified Framework for Embodied Navigation Integrating Perception, Planning, and Prediction**](https://arxiv.org/pdf/2503.18525) | ICCV 2025 | [Page](https://yvfengzhong.github.io/RoboTron-Nav/) | [Github](https://github.com/EmbodiedAI-RoboTron/RoboTron-Nav) |
| [**FLAME: Learning to Navigate with Multimodal LLM in Urban Environments**](https://arxiv.org/pdf/2408.11051) | AAAI 2025 | [Page](https://flame-sjtu.github.io/) | [Github](https://github.com/xyz9911/FLAME) |
| [**WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation**](https://arxiv.org/pdf/2503.02247) | IROS 2025 | [Page](https://b0b8k1ng.github.io/WMNav/) | [Github](https://github.com/B0B8K1ng/WMNavigation) |
| [**SmartWay: Enhanced Waypoint Prediction and Backtracking for Zero-Shot Vision-and-Language Navigation**](https://arxiv.org/pdf/2503.10069) | IROS 2025 | [Page](https://sxyxs.github.io/smartway/) | - |
| [**NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation**](https://arxiv.org/pdf/2402.15852) | RSS 2024 | [Page](https://pku-epic.github.io/NaVid/) | [Github](https://github.com/jzhzhang/NaVid-VLN-CE) |
| [**LLaDA: Driving Everywhere with Large Language Model Policy Adaptation**](https://arxiv.org/pdf/2402.05932) | CVPR 2024 | [Page](https://boyiliee.github.io/llada/) | [Github](https://github.com/Boyiliee/LLaDA-AV) |
| [**Adaptive Zone-aware Hierarchical Planner**](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_Adaptive_Zone-Aware_Hierarchical_Planner_for_Vision-Language_Navigation_CVPR_2023_paper.pdf) | CVPR 2024 | - | [Github](https://github.com/chengaopro/AZHP) |




## 📊 Benchmarks & Datasets

<!-- Detection -->
<!-- Segmentation -->
<!-- SLAM -->
<!-- Depth Estimation -->
<!-- 3D -->
### Perception
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D**](https://arxiv.org/pdf/2503.22976) | NeurIPS 2025 | [Page](https://logosroboticsgroup.github.io/SPAR/) | [Github](https://github.com/LogosRoboticsGroup/SPAR) |
| [**Large-scale Dataset and Benchmark for Egocentric Robot Perception and Navigation in Crowded and Unstructured Environments**](https://arxiv.org/pdf/2408.15503) | CVPR 2025 | - | [Github](https://github.com/suhaisheng/RoboSense) |
| [**RCP-Bench: Robust Collaborative Perception Framework**](https://openaccess.thecvf.com/content/CVPR2025/papers/Du_RCP-Bench_Benchmarking_Robustness_for_Collaborative_Perception_Under_Diverse_Corruptions_CVPR_2025_paper.pdf) | CVPR 2025 | - | [Github](https://github.com/LuckyDush/RCP-Bench) |
| [**Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models**](https://arxiv.org/pdf/2409.17146v1) | CVPR 2025 | [Page](https://playground.allenai.org/?model=mm-olmo-uber-model-v4-synthetic) | [Github](https://github.com/allenai/molmo) |
| [**TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation**](https://arxiv.org/pdf/2505.10696) | IROS 2025 | [Page](https://tartanair.org/tartanground/) | [Github](https://github.com/castacks/tartanairpy) |
| [**HRIBench: Benchmarking Vision-Language Models for Real-Time Human Perception in Human-Robot Interaction**](https://arxiv.org/pdf/2506.20566) | ISER 2025 | - | [Github](https://github.com/interaction-lab/HRIBench) |
| [**EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI**](https://arxiv.org/pdf/2312.16170) | CVPR 2024 | [Page](https://tai-wang.github.io/embodiedscan/) | [Github](https://github.com/InternRobotics/EmbodiedScan) |
| [**MCD: Diverse Large-Scale Multi-Campus Dataset for Robot Perception**](https://arxiv.org/pdf/2403.11496) | CVPR 2024 | [Page](https://mcdviral.github.io/) | [Github](https://github.com/mcdviral/mcdviral.github.io) |
| [**MMScan: A Multi-Modal 3D Scene Dataset with Hierarchical Grounded Language Annotations**](https://tai-wang.github.io/mmscan/static/paper/MMScan.pdf) | NeurIPS 2024 | [Page](https://tai-wang.github.io/mmscan/) | [Github](https://github.com/InternRobotics/EmbodiedScan) |
| [**Tiny Robotics Dataset and Benchmark for Continual Object Detection**](https://arxiv.org/pdf/2409.16215) | Arxiv 2024 | - | [Github](https://github.com/pastifra/TiROD_code) |
| [**Perception Test: A Diagnostic Benchmark for Multimodal Video Models**](https://arxiv.org/pdf/2305.13786) | NeurIPS 2023 | [Page](https://perception-test-challenge.github.io/) | [Github](https://github.com/google-deepmind/perception_test) |
| [**Robo3D: Towards Robust and Reliable 3D Perception against Corruptions**](https://arxiv.org/pdf/2303.17597) | ICCV 2023 | [Page](https://ldkong.com/Robo3D) | [Github](https://github.com/worldbench/robo3d) |
| [**Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous Driving**](https://arxiv.org/pdf/2303.11040) | CVPR 2023 | - | [Github](https://github.com/thu-ml/3D_Corruptions_AD) |
| [**Ost-bench: Evaluating The Capabilities Of Mllms In Online Spatio-temporal Scene Understanding**](https://arxiv.org/pdf/2507.07984) | Arxiv 2025 | [Page](https://rbler1234.github.io/OSTBench.github.io/) | [Github](https://github.com/OpenRobotLab/OST-Bench) |
| [**A Unified Perception Benchmark for Capacitive Proximity Sensing Towards Safe Human-Robot Collaboration (HRC)**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561224) | ICRA 2021 | - | - |
| [**JRDB: A Dataset and Benchmark of Egocentric Robot Visual Perception of Humans in Built Environments**](https://arxiv.org/pdf/1910.11792) | TPAMI 2019 | [Page](https://jrdb.erc.monash.edu/) | - |
| [**BOP: Benchmark for 6D Object Pose Estimation**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tomas_Hodan_PESTO_6D_Object_ECCV_2018_paper.pdf) | ECCV 2018 | [Page](https://bop.felk.cvut.cz/home/) | - |






<!-- 
| [****]() | ICML 2025 | [Page]() | [Github]() |
-->
<!-- Spatial Reasoning -->
<!-- Question Answering -->
### Reasoning
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics**](https://arxiv.org/pdf/2506.04308) | NeurIPS 2025 | [Page](https://zhoues.github.io/RoboRefer/) | [Github](https://github.com/Zhoues/RoboRefer) |
| [**Robo2VLM: Visual Question Answering from Large-Scale In-the-Wild Robot Manipulation Datasets**](https://arxiv.org/pdf/2505.15517) | NeurIPS 2025 | [Page](https://berkeleyautomation.github.io/robo2vlm/) | [Github](https://github.com/KeplerC/robo2VLM) |
| [**NuScenes-SpatialQA: A Spatial Understanding and Reasoning Benchmark for Vision-Language Models in Autonomous Driving**](https://arxiv.org/pdf/2504.03164) | ICCV 2025 | [Page](https://taco-group.github.io/NuScenes-SpatialQA/) | [Github](https://github.com/taco-group/NuScenes-SpatialQA) |
| [**Beyond the Destination: A Novel Benchmark for Exploration-Aware Embodied Question Answering**](https://arxiv.org/pdf/2503.11117) | ICCV 2025 | [Page](https://hcplab-sysu.github.io/EXPRESS-Bench/) | [Github](https://github.com/HCPLab-SYSU/EXPRESS-Bench) |
| [**Embodied Reasoning QA Evaluation Dataset**](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf) | Technical report | [Page](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/) | [Github](https://github.com/embodiedreasoning/ERQA) |
| [**ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation**](https://arxiv.org/pdf/2409.13682) | ICRA 2025 | [Page](https://nvidia-ai-iot.github.io/remembr/) | [Github](https://github.com/NVIDIA-AI-IOT/remembr) |
| [**SpatialBot: Precise Spatial Understanding with Vision Language Models**](https://arxiv.org/pdf/2406.13642) | ICRA 2025 | - | [Github](https://github.com/BAAI-DCAI/SpatialBot) |
| [**Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces**](https://arxiv.org/pdf/2412.14171) | CVPR 2025 | [Page](https://vision-x-nyu.github.io/thinking-in-space.github.io/) | [Github](https://github.com/vision-x-nyu/thinking-in-space) |
| [**PhysBench Benchmarking and Enhancing VLMs for Physical World Understanding**](https://arxiv.org/pdf/2501.16411) | ICLR 2025 | [Page](https://physbench.github.io/) | [Github](https://github.com/USC-GVL/PhysBench) |
| [**OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models**](https://arxiv.org/pdf/2506.03135) | Arxiv | [Page](https://qizekun.github.io/omnispatial/) | [Github](https://github.com/qizekun/OmniSpatial) |
| [**MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence**](https://arxiv.org/pdf/2505.23764) | Arxiv 2025 | [Page](https://runsenxu.com/projects/MMSI_Bench/) | [Github](https://github.com/InternRobotics/MMSI-Bench) |
| [**ViLaSR: Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing**](https://arxiv.org/pdf/2506.09965) | Arxiv 2025 | - | [Github](https://github.com/AntResearchNLP/ViLaSR) |
| [**SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models**](https://arxiv.org/pdf/2406.01584) | NeurIPS 2024 | [Page](https://www.anjiecheng.me/SpatialRGPT) | [Github](https://github.com/AnjieCheng/SpatialRGPT) |
| [**OpenEQA: Embodied Question Answering in the Era of Foundation Models**](https://open-eqa.github.io/assets/pdfs/paper.pdf) | CVPR 2024 | [Page](https://open-eqa.github.io/) | [Github](https://github.com/facebookresearch/open-eqa) |
| [**EQA-MX: Embodied Question Answering using Multimodal Expression**](https://openreview.net/pdf?id=7gUrYE50Rb) | ICLR 2024 | - | [Github](https://github.com/mmiakashs/eqa-mx) |
| [**RoboVQA: Multimodal Long-Horizon Reasoning for Robotics**](https://arxiv.org/pdf/2311.00899) | ICRA 2024 | [Page](https://robovqa.github.io/) | [Github](https://github.com/google-deepmind/robovqa) |
| [**EgoTaskQA: Understanding Human Tasks in Egocentric Videos**](https://arxiv.org/pdf/2210.03929) | NeurIPS 2022 | [Page](https://sites.google.com/view/egotaskqa) | [Github](https://github.com/Buzz-Beater/EgoTaskQA) |



### Planning
<!-- Planning -->
<!-- Decision Making -->
<!-- Agent -->
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents**](https://arxiv.org/pdf/2502.09560) | ICML 2025 | [Page](https://embodiedbench.github.io/) | [Github](https://github.com/EmbodiedBench/EmbodiedBench)
| [**PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-agent Tasks**](https://arxiv.org/pdf/2411.00081) | ICLR 2025 | [Page](https://aihabitat.org/partnr) | [Github](https://github.com/facebookresearch/partnr-planner) |
| [**OWMM-Agent: Open World Mobile Manipulation With Multi-modal Agentic Data Synthesis**](https://arxiv.org/pdf/2506.04217) | NeurIPS 2025 | - | [Github](https://github.com/HHYHRHY/OWMM-Agent) |
| [**ET-Plan-Bench: Embodied Task-level Planning Benchmark Towards Spatial-Temporal Cognition with Foundation Models**](https://arxiv.org/pdf/2410.14682) | IROS 2025 | - | [Github](https://github.com/ET-Plan-Bench/ET-Plan-Bench) |
| [**EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents**](https://arxiv.org/pdf/2501.11858) | Arxiv 2025 | [Page](https://embodiedeval.github.io/) | [Github](https://github.com/thunlp/EmbodiedEval) |
| [**WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning**](https://arxiv.org/pdf/2506.04363) | Arxiv 2025 | [Page](https://worldprediction.github.io/) | [Github](https://github.com/fairinternal/WorldPrediction) |
| [**Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making**](https://arxiv.org/pdf/2410.07166) | NeurIPS 2024 | [Page](https://embodied-agent-interface.github.io/) | [Github](https://github.com/embodied-agent-interface/embodied-agent-interface) |
| [**Large Language Models as Generalizable Policies for Embodied Tasks**](https://arxiv.org/pdf/2310.17722) | ICLR 2024 | [Page](https://llm-rl.github.io/) | [Github](https://github.com/apple/ml-llarp) |
| [**LoTa-Bench: Benchmarking Language-oriented Task Planners for Embodied Agents**](https://arxiv.org/pdf/2402.08178) | ICLR 2024 | [Page](https://choi-jaewoo.github.io/LoTa-Bench/) | [Github](https://github.com/lbaa2022/LLMTaskPlanning) |
| [**HAZARD Challenge: Embodied Decision Making in Dynamically Changing Environments**](https://arxiv.org/pdf/2401.12975) | ICLR 2024 | [Page](https://embodied-agi.cs.umass.edu/hazard/) | [Github](https://github.com/UMass-Embodied-AGI/HAZARD) |
| [**MFE-ETP: A Comprehensive Evaluation Benchmark for Multi-modal Foundation Models on Embodied Task Planning**](https://arxiv.org/pdf/2407.05047) | Arxiv 2024 | [Page](https://mfe-etp.github.io/) | [Github](https://github.com/TJURLLAB-EAI/MFE-ETP) |
| [**EgoPlan-Bench2: A Benchmark for Multimodal Large Language Model Planning in Real-World Scenarios**](https://arxiv.org/pdf/2412.04447) | Arxiv 2024 | [Page](https://qiulu66.github.io/egoplanbench2/) | [Github](https://github.com/qiulu66/EgoPlan-Bench2/) |
| [**SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents**](https://arxiv.org/pdf/2412.13178) | Arxiv 2024 | - | [Github](https://github.com/shengyin1224/SafeAgentBench) |
| [**EgoPlan-Bench: Benchmarking Multimodal Large Language Models for Human-Level Planning**](https://arxiv.org/pdf/2312.06722) | Arxiv 2023 | [Page](https://chenyi99.github.io/ego_plan/) | [Github](https://github.com/ChenYi99/EgoPlan) |
| [**Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots**](https://arxiv.org/pdf/2310.13724) | Arxiv 2023 | [Page](https://aihabitat.org/habitat3/) | [Github](https://github.com/facebookresearch/habitat-lab) |
| [**Habitat 2.0: Training Home Assistants to Rearrange their Habitat**](https://arxiv.org/pdf/2106.14405) | NeurIPS 2021 | [Page](https://sites.google.com/view/habitat2) | [Github](https://github.com/facebookresearch/habitat-lab) |
| [**ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks**](https://arxiv.org/pdf/1912.01734) | CVPR 2020 | [Page](https://askforalfred.com/) | [Github](https://github.com/askforalfred/alfred) |
| [**Habitat: A Platform for Embodied AI Research**](https://arxiv.org/pdf/1904.01201) | ICCV 2019 | [Page](https://aihabitat.org/) | [Github](https://github.com/facebookresearch/habitat-lab) |


### Control

<!-- Grasp -->
<!-- Dexterous Hand -->
<!-- Pick & Place -->
#### Manipulation
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation**](https://arxiv.org/pdf/2412.13877) | RSS 2025 | [Page](https://x-humanoid-robomind.github.io/) | [Github](https://github.com/x-humanoid-robomind/x-humanoid-robomind.github.io) |
| [**ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI**](https://arxiv.org/pdf/2410.00425) | RSS 2025 | [Page](https://www.maniskill.ai/) | [Github](https://github.com/haosulab/ManiSkill) |
| [**RoboVerse: Towards a Unified Platform, Dataset and Benchmark for Scalable and Generalizable Robot Learning**](https://arxiv.org/pdf/2504.18904) | RSS 2025 | [Page](https://roboverseorg.github.io/) | [Github](https://github.com/RoboVerseOrg/RoboVerse) |
| [**Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation**](https://jianglongye.com/dex1b/static/dex1b.pdf) | RSS 2025 | [Page](https://jianglongye.com/dex1b/) | - |
| [**ManipBench: Benchmarking Vision-Language Models for Low-Level Robot Manipulation**](https://arxiv.org/pdf/2505.09698) | CoRL 2025 | [Page](https://manipbench.github.io/) | [Github](https://github.com/slurm-lab-usc/ManipBench-Real-Robot-question) |
| [**ManiFeel: Benchmarking and Understanding Visuotactile Manipulation Policy Learning**](https://arxiv.org/pdf/2505.18472) | CoRL 2025 | [Page](https://zhengtongxu.github.io/manifeel-website/) | - |
| [**AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems**](https://arxiv.org/pdf/2503.06669) | IROS 2025 | [Page](https://agibot-world.com/) | [Github](https://github.com/OpenDriveLab/Agibot-World) |
| [**Gembench: Towards Generalizable Vision-Language Robotic Manipulation: A Benchmark and LLM-guided 3D Policy**](https://arxiv.org/pdf/2410.01345) | ICRA 2025 | [Page](https://www.di.ens.fr/willow/research/gembench/) | [Github](https://github.com/vlc-robot/robot-3dlotus) |
| [**RoboCerebra：A Large-scale Benchmark for Long-Horizon Robotic Manipulation Evaluation**](https://www.arxiv.org/pdf/2506.06677) | NeurIPS 2025 | [Page](https://robocerebra.github.io/) | [Github](https://github.com/qiuboxiang/RoboCerebra) |
| [**VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks**](https://arxiv.org/pdf/2412.18194) | ICCV 2025 | [Page](https://vlabench.github.io/) | [Github](https://github.com/OpenMOSS/VLABench) |
| [**DexH2R: A Benchmark for Dynamic Dexterous Grasping in Human-to-Robot Handover**](https://arxiv.org/pdf/2506.23152) | ICCV 2025 | [Page](https://dexh2r.github.io/) | [Github](https://github.com/4DVLab/DexH2R) |
| [**ManiSkill-HAB: A Benchmark for Low-Level Manipulation in Home Rearrangement Tasks**](https://arxiv.org/pdf/2412.13211) | ICLR 2025 | [Page](https://arth-shukla.github.io/mshab/) | [Github](https://github.com/arth-shukla/mshab) |
| **GENESIS: A generative world for general-purpose robotics & embodied AI learning** | - | [Page](https://genesis-world.readthedocs.io/en/latest/) | [Github](https://github.com/Genesis-Embodied-AI/Genesis) |
| [**RoboCAS: A Benchmark for Robotic Manipulation in Complex Object Arrangement Scenarios**](https://arxiv.org/pdf/2407.06951v1) | NeurIPS 2024 | - | [Github](https://github.com/notFoundThisPerson/RoboCAS-v0) |
| [**Towards Diverse Behaviors: A Benchmark for Imitation Learning with Human Demonstrations**](https://arxiv.org/pdf/2402.14606) | ICLR 2024 | [Page](https://alrhub.github.io/d3il-website/) | [Github](https://github.com/ALRhub/d3il) |
| [**FetchBench: A Simulation Benchmark for Robot Fetching**](https://arxiv.org/pdf/2406.11793) | CoRL 2024 | - | [Github](https://github.com/princeton-vl/FetchBench-CORL2024) |
| [**DexGraspNet 2.0: Learning Generative Dexterous Grasping in Large-scale Synthetic Cluttered Scenes**](https://arxiv.org/pdf/2410.23004) | CoRL 2024 | [Page](https://pku-epic.github.io/DexGraspNet2.0/) | [Github](https://github.com/PKU-EPIC/DexGraspNet2) |
| [**Open X-Embodiment: Robotic Learning Datasets and RT-X Models**](https://arxiv.org/pdf/2310.08864) | ICRA 2024 | [Page](https://robotics-transformer-x.github.io/) | [Github](https://github.com/google-deepmind/open_x_embodiment) |
| [**RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot**](https://arxiv.org/pdf/2307.00595) | ICRA 2024 | [Page](https://rh20t.github.io/) | [Github](https://github.com/rh20t/rh20t_api) |
| [**SceneReplica: Benchmarking Real-World Robot Manipulation by Creating Replicable Scenes**](https://arxiv.org/pdf/2306.15620) | ICRA 2024 | [Page](https://irvlutd.github.io/SceneReplica/) | [Github](https://github.com/IRVLUTD/SceneReplica) |
| [**Grasp-Anything: Large-scale Grasp Dataset from Foundation Models**](https://arxiv.org/pdf/2309.09818) | ICRA 2024 | [Page](https://airvlab.github.io/grasp-anything/) | [Github](https://github.com/Fsoft-AIC/Grasp-Anything) |
| [**RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots**](https://robocasa.ai/assets/robocasa_rss24.pdf) | RSS 2024 | [Page](https://robocasa.ai/) | [Github](https://github.com/robocasa/robocasa) |
| [**SimplerEnv: Simulated Manipulation Policy Evaluation Environments for Real Robot Setups**](https://arxiv.org/pdf/2405.05941) | CoRL 2024 | [Page](https://simpler-env.github.io/) | [Github](https://github.com/simpler-env/SimplerEnv) |
| [**HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation**](https://arxiv.org/pdf/2403.10506) | Arxiv 2024 | [Page](https://humanoid-bench.github.io/) | [Github](https://github.com/carlosferrazza/humanoid-bench) |
| [**HomeRobot: Open-Vocabulary Mobile Manipulation**](https://arxiv.org/pdf/2306.11565) | CoRL 2023 | [Page](https://ovmm.github.io/) | [Github](https://github.com/facebookresearch/home-robot) |
| [**DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation**](https://arxiv.org/pdf/2210.02697) | ICRA 2023 | [Page](https://pku-epic.github.io/DexGraspNet/) | [Github](https://github.com/PKU-EPIC/DexGraspNet) |
| [**LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning**](https://arxiv.org/pdf/2306.03310) | NeurIPS 2023 | [Page](https://libero-project.github.io/intro.html) | [Github](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| [**RoboHive: A Unified Framework for Robot Learning**](https://arxiv.org/pdf/2310.06828) | NeurIPS 2023 | [Page](https://sites.google.com/view/robohive) | [Github](https://github.com/vikashplus/robohive) |
| [**ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes**](https://arxiv.org/pdf/2304.04321) | ICCV 2023 | [Page](https://arnold-benchmark.github.io/) | [Github](https://github.com/arnold-benchmark/arnold) |
| [**ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills**](https://arxiv.org/pdf/2302.04659) | ICLR 2023 | [Page](https://www.maniskill.ai/) | [Github](https://github.com/haosulab/ManiSkill) |
| [**DaXBench: Benchmarking Deformable Object Manipulation with Differentiable Physics**](https://arxiv.org/pdf/2210.13066) | ICLR 2023 | [Page](https://daxbench.github.io/) | [Github](https://github.com/AdaCompNUS/DaXBench) |
| [**VIMA: General Robot Manipulation with Multimodal Prompts**](https://vimalabs.github.io/assets/vima_paper.pdf) | ICML 2023 | [Page](https://vimalabs.github.io/) | [Github](https://github.com/vimalabs/VIMA) |
| [**Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments**](https://arxiv.org/pdf/2301.04195) | RA-L 2023 | [Page](https://isaac-orbit.github.io/) | [Github](https://github.com/isaac-sim/IsaacLab) |
| [**CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks**](https://arxiv.org/pdf/2112.03227) | RA-L 2022 | [Page](http://calvin.cs.uni-freiburg.de/) | [Github](https://github.com/mees/calvin) |
| [**BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation**](https://arxiv.org/pdf/2403.09227) | CoRL 2022 | [Page](https://behavior.stanford.edu/) | [Github](https://github.com/StanfordVL/BEHAVIOR-1K) |
| [**Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets**](https://arxiv.org/pdf/2109.13396) | RSS 2022 | [Page](https://sites.google.com/view/bridgedata) | [Github](https://github.com/yanlai00/bridge_data_imitation_learning) |
| [**What Matters in Learning from Offline Human Demonstrations for Robot Manipulation**](https://arxiv.org/pdf/2108.03298) | CoRL 2021 | [Page](https://robomimic.github.io/) | [Github](https://github.com/ARISE-Initiative/robomimic) |
| [**PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics**](https://arxiv.org/pdf/2104.03311) | ICLR 2021 | [Page](https://plasticinelab.csail.mit.edu/) | [Github](https://github.com/hzaskywalker/PlasticineLab) |
| [**ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations**](https://arxiv.org/pdf/2107.14483) | NeurIPS 2021 | - | [Github](https://github.com/haosulab/ManiSkill) |
| [**DexYCB: A Benchmark for Capturing Hand Grasping of Objects**](https://arxiv.org/pdf/2104.04631) | CVPR 2021 | [Page](https://dex-ycb.github.io/) | [Github](https://github.com/NVlabs/dex-ycb-toolkit) |
| [**RLBench: The Robot Learning Benchmark & Learning Environment**](https://arxiv.org/pdf/1909.12271) | RA-L 2020 | [Page](https://sites.google.com/view/rlbench) | [Github](https://github.com/stepjam/RLBench) |
| [**Benchmarking In-Hand Manipulation**](https://ieeexplore.ieee.org/document/8950086) | RA-L 2020 | [Page](https://robot-learning.cs.utah.edu/project/benchmarking_in_hand_manipulation) | - |
| [**GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf) | CVPR 2020 | [Page](https://graspnet.net/) | [Github](https://github.com/graspnet/graspnet-baseline) |
| [**robosuite: A Modular Simulation Framework and Benchmark for Robot Learning**](https://arxiv.org/pdf/2009.12293) | Arxiv 2020 | [Page](https://robosuite.ai/) | [Github](https://github.com/ARISE-Initiative/robosuite) |



<!-- Autonomous driving -->
<!-- Vision-Language-Navigation -->
#### Navigation
|  Title  |   Venue  |   Website   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives**](https://arxiv.org/pdf/2501.04003) | ICCV 2025 | [Page](https://drive-bench.github.io/) | [Github](https://github.com/worldbench/drivebench) |
| [**Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method**](https://arxiv.org/pdf/2412.09082) | CVPR 2025 | [Page](https://hcplab-sysu.github.io/LH-VLN/) | [Github](https://github.com/HCPLab-SYSU/LH-VLN) |
| [**OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning**](https://arxiv.org/pdf/2405.01533) | CVPR 2025 | [Page](https://research.nvidia.com/labs/lpr/publication/wang2024omnidrive/) | [Github](https://github.com/NVlabs/OmniDrive) |
| [**CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving**](https://arxiv.org/pdf/2408.10845) | WACV 2025 | [Page](https://turingmotors.github.io/covla-ad/) | - |
| [**FlightBench: Benchmarking Learning-based Methods for Ego-vision-based Quadrotors Navigation**](https://arxiv.org/pdf/2406.05687) | RA-L 2025 | [Page](https://thu-uav.github.io/FlightBench/) | [Github](https://github.com/thu-uav/FlightBench) |
| [**Memory-Maze: Scenario Driven Benchmark and Visual Language Navigation Model for Guiding Blind People**](https://arxiv.org/pdf/2405.07060) | RA-L 2025 | - | - |
| [**GND: Global Navigation Dataset with Multi-Modal Perception and Multi-Category Traversability in Outdoor Campus Environments**](https://arxiv.org/pdf/2409.14262) | ICRA 2025 | [Page](https://cs.gmu.edu/~xiao/Research/GND/) | [Github](https://github.com/jingGM/GND) |
| [**Language Prompt for Autonomous Driving**](https://arxiv.org/pdf/2309.04379) | AAAI 2025 | - | [Github](https://github.com/wudongming97/Prompt4Driving) |
| [**Human-Aware Vision-and-Language Navigation: Bridging Simulation to Reality with Dynamic Human Interactions**](https://arxiv.org/pdf/2406.19236v1) | NeurIPS 2024 | [Page](https://lpercc.github.io/HA3D_simulator/) | [Github](https://github.com/lpercc/HA3D_simulator) |
| [**HM3D-OVON: A Dataset and Benchmark for Open-Vocabulary Object Goal Navigation**](https://arxiv.org/pdf/2409.14296) | IROS 2024 | [Page](https://naoki.io/portfolio/ovon.html) | [Github](https://github.com/naokiyokoyama/ovon) |
| [**Generalized Predictive Model for Autonomous Driving**](https://arxiv.org/pdf/2403.09630) | CVPR 2024 | - | [Github](https://github.com/OpenDriveLab/DriveAGI) |
| [**GOAT-Bench: A Benchmark for Multi-modal Lifelong Navigation**](https://arxiv.org/pdf/2404.06609) | CVPR 2024 | [Page](https://mukulkhanna.github.io/goat-bench/) | [Github](https://github.com/Ram81/goat-bench) |
| [**Rank2Tell: A Multimodal Driving Dataset for Joint Importance Ranking and Reasoning**](https://arxiv.org/pdf/2309.06597) | WACV 2024 | [Page](https://usa.honda-ri.com/rank2tell) | - |
| [**BenchNav: Simulation Platform for Benchmarking Off-road Navigation Algorithms with Probabilistic Traversability**](https://arxiv.org/pdf/2405.13318) | Arxiv 2024 | - | [Github](https://github.com/masafumiendo/benchnav) |
| [**Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology**](https://arxiv.org/pdf/2410.07087) | Arxiv 2024 | [Page](https://prince687028.github.io/OpenUAV/) | [Github](https://github.com/prince687028/TravelUAV) |
| [**Toward Human-Like Social Robot Navigation: A Large-Scale, Multi-Modal, Social Human Navigation Dataset**](https://cs.gmu.edu/~xiao/papers/musohu.pdf) | IROS 2023 | [Page](https://cs.gmu.edu/~xiao/Research/MuSoHu/) | [Github](https://github.com/RobotiXX/MuSoHu-data-collection) |
| [**Benchmarking Visual Localization for Autonomous Navigation**](https://arxiv.org/pdf/2203.13048) | WACV 2023 | [Page](https://lasuomela.github.io/carla_vloc_benchmark/) | [Github](https://github.com/lasuomela/carla_vloc_benchmark) |
| [**GOAT: GO to Any Thing**](https://arxiv.org/pdf/2311.06430) | Arxiv 2023 | [Page](https://theophilegervet.github.io/projects/goat/) | [Github](https://github.com/facebookresearch/home-robot) |
| [**RobustNav: Towards Benchmarking Robustness in Embodied Navigation**](https://arxiv.org/pdf/2106.04531) | ICCV 2021 | [Page](https://prior.allenai.org/projects/robustnav) | [Github](https://github.com/allenai/robustnav) |
| [**MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation**](https://arxiv.org/pdf/2012.03912) | NeurIPS 2020 | [Page](https://shivanshpatel35.github.io/multi-ON/) | [Github](https://github.com/saimwani/multiON) |
| [**The RobotSlang Benchmark: Dialog-guided Robot Localization and Navigation**](https://arxiv.org/pdf/2010.12639) | CoRL 2020 | [Page](https://umrobotslang.github.io/) | [Github](https://github.com/MichiganCOG/RobotSlangBenchmark) |
| [**Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments**](https://arxiv.org/pdf/2004.02857) | ECCV 2020 | [Page](https://jacobkrantz.github.io/vlnce/) | [Github](https://github.com/jacobkrantz/VLN-CE) |
| [**Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding**](https://arxiv.org/pdf/2010.07954) | EMNLP 2020 | [Page](https://jacobkrantz.github.io/vlnce/) | [Github](https://github.com/jacobkrantz/VLN-CE) |
| [**Explainable Object-induced Action Decision for Autonomous Vehicles**](https://arxiv.org/pdf/2003.09405) | CVPR 2020 | [Page](https://twizwei.github.io/bddoia_project/) | [Github](https://github.com/Twizwei/bddoia_project) |
| [**nuScenes: A multimodal dataset for autonomous driving**](https://arxiv.org/pdf/1903.11027) | CVPR 2020 | [Page](https://www.nuscenes.org/) | - |
| [**The apolloscape open dataset for autonomous driving and its application**](https://arxiv.org/pdf/1803.06184) | TPAMI 2019 | [Page](https://apolloscape.auto/) | [Github](https://github.com/ApolloScapeAuto/dataset-api) |
| [**Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments**](https://arxiv.org/pdf/1711.07280) | CVPR 2018 | [Page](https://bringmeaspoon.org/) | - |



<!-- 
## Others
| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **IMAD** | [IMAD: IMage-Augmented multi-modal Dialogue](https://arxiv.org/pdf/2305.10512.pdf) | [Link](https://github.com/VityaVitalich/IMAD) | Multimodal dialogue dataset|
| **Video-ChatGPT** | [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://arxiv.org/pdf/2306.05424.pdf) | [Link](https://github.com/mbzuai-oryx/Video-ChatGPT#quantitative-evaluation-bar_chart) | A quantitative evaluation framework for video-based dialogue models | -->
