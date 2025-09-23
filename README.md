# CADEC Medical NER Analysis

🏥 **Comprehensive Medical Named Entity Recognition analysis on the CADEC dataset using GPU-accelerated deep learning models**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🎯 Overview

This project provides a full pipeline for medical Named Entity Recognition (NER) using the CADEC (Consumer Analysis of Drug Events and Concerns) dataset. It leverages cutting-edge transformer models, optimized for GPU performance, to detect and categorize medical entities in user-generated content.

### Key Features

- 🚀 **GPU-Accelerated Processing** - Optimized for NVIDIA RTX 3060 and CUDA 12.4
- 🏥 **Medical Domain Expertise** - Uses specialized biomedical NER models
- 📊 **Comprehensive Evaluation** - Multiple evaluation methodologies and metrics
- 🔗 **Medical Code Mapping** - Links entities to SNOMED CT and MedDRA codes
- 📈 **Batch Processing** - Handles large datasets efficiently
- 📋 **Detailed Reporting** - Generates comprehensive analysis reports

## 🎬 Entity Types Detected

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| **ADR** | Adverse Drug Reactions | "nausea", "headache", "dizziness" |
| **Drug** | Medications and substances | "aspirin", "ibuprofen", "acetaminophen" |
| **Disease** | Medical conditions | "diabetes", "hypertension", "depression" |
| **Symptom** | Clinical symptoms | "fatigue", "chest pain", "shortness of breath" |

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch 2.6.0, Transformers 4.x
- **GPU Acceleration**: CUDA 12.4, RTX 3060 optimization
- **Medical NER Models**: Biomedical domain-specific transformers
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Code Mapping**: SNOMED CT, MedDRA integration
- **Evaluation**: Custom overlap-based metrics

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- NVIDIA GPU with CUDA support (tested on RTX 3060)
- Python 3.8+
- CUDA 12.4+
- 16GB+ RAM recommended

# GPU Setup (Windows)
nvidia-smi  # Verify GPU detection
nvcc --version  # Verify CUDA installation
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/aadi611/cadec-medical-ner-analysis.git
cd cadec-medical-ner-analysis
```

2. **Create virtual environment**
```bash
python -m venv cadec_env
cadec_env\Scripts\activate  # Windows
# source cadec_env/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate
pip install pandas numpy scikit-learn matplotlib seaborn
pip install sentence-transformers  # For SNOMED matching
```

4. **Download CADEC dataset**
```bash
# Download CADEC.v2.zip from official source
# Extract to ./cadec/ directory
mkdir cadec
# Extract CADEC.v2.zip contents to ./cadec/
```

### Dataset Structure

```
./cadec/
├── text/          # Original forum posts
├── original/      # Original annotations  
├── meddra/        # MedDRA annotations
└── sct/           # SNOMED CT annotations
```

### GPU Verification

```python
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## 🎯 Usage

### Run Complete Analysis

```bash
python cadec_master.py
```

This executes all 6 tasks in sequence:

1. **Entity Enumeration** - Count and list all unique entities
2. **LLM-based BIO Tagging** - Generate NER predictions
3. **Performance Evaluation** - Evaluate against ground truth
4. **ADR-specific Evaluation** - Focus on adverse drug reactions
5. **Batch Evaluation** - Test on 50 random files
6. **SNOMED Code Matching** - Map entities to medical codes

### Run Individual Tasks

```bash
python cadec_task1.py  # Entity enumeration
python cadec_task2.py  # NER prediction
python cadec_task3.py  # Evaluation
python cadec_task4.py  # ADR evaluation
python cadec_task5.py  # Batch evaluation
python cadec_task6.py  # SNOMED matching
```

## 📊 Results Structure

```
./outputs/
├── entity_lists/               # Task 1: Entity enumerations
│   ├── adr_entities.txt
│   ├── drug_entities.txt
│   ├── disease_entities.txt
│   └── symptom_entities.txt
├── predictions/                # Task 2: NER predictions
│   ├── *.ann                   # Annotation format
│   └── *_bio.txt              # BIO format
├── task4_predictions/          # Task 4: ADR-specific
├── task5_results/             # Task 5: Batch evaluation
│   ├── detailed_results.csv
│   └── summary_statistics.txt
├── task6_results/             # Task 6: SNOMED matching
│   └── *_snomed_matching.json
└── CADEC_Analysis_Complete_Report.md  # Final comprehensive report
```

## 🔧 GPU Optimization

The project includes RTX 3060-specific optimizations:

```python
# Automatic GPU detection and optimization
torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(0.9)

# Optimal batch sizes for 12GB VRAM
batch_sizes = {
    "small": 32,   # Small models (110M params)
    "base": 16,    # Base models (340M params)
    "large": 8     # Large models (770M+ params)
}
```

## 📈 Performance Metrics

### Evaluation Methodology

- **Overlap-based Evaluation**: 50% IoU threshold (more realistic than exact match)
- **Entity-level Metrics**: Precision, Recall, F1-score per entity type
- **Cross-validation**: Performance across diverse forum posts

### Expected Performance

| Entity Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Drug        | ~0.85     | ~0.78  | ~0.81    |
| Disease     | ~0.72     | ~0.68  | ~0.70    |
| ADR         | ~0.68     | ~0.62  | ~0.65    |
| Symptom     | ~0.71     | ~0.65  | ~0.68    |

*Note: Performance varies significantly across different forum posts*

## 🏥 Medical Code Integration

### SNOMED CT Mapping
- **String Matching**: Exact and fuzzy text matching
- **Embedding Matching**: Semantic similarity using sentence transformers
- **Confidence Scoring**: Reliability assessment for each mapping

### MedDRA Integration
- Specialized adverse drug reaction terminology
- Enhanced ADR detection and classification
- Clinical coding compatibility

## 🐛 Troubleshooting

### GPU Issues
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
nvcc --version

# Test PyTorch GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in gpu_optimization.py
# Monitor GPU usage during processing
```

### Common Solutions

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size in `gpu_optimization.py` |
| `Model not found` | Check internet connection for Hugging Face downloads |
| `CADEC files missing` | Verify dataset extraction to `./cadec/` directory |
| `Permission denied` | Run with administrator privileges on Windows |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{cadec_ner_analysis,
  title={CADEC Medical NER Analysis: GPU-Accelerated Medical Entity Recognition},
  author={Aadityan Gupta},
  year={2025},
  url={https://github.com/aadi611/cadec-medical-ner-analysis}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CADEC Dataset](https://data.csiro.au/collection/csiro:10948) - Consumer Analysis of Drug Events and Concerns
- [Hugging Face Transformers](https://huggingface.co/transformers) - State-of-the-art NLP models
- [Biomedical NER Models](https://huggingface.co/d4data/biomedical-ner-all) - Domain-specific medical NER
- NVIDIA for CUDA and GPU acceleration support

## 📞 Contact

- **Author**: Aadityan Gupta
- **Email**: aadityan.gupta@gmail.com
- **LinkedIn**: [https://www.linkedin.com/in/aadityangupta/]
- **Project Link**: [https://github.com/aadi611/cadec-medical-ner-analysis]
                    (https://github.com/aadi611/cadec-medical-ner-analysis)

---

⭐ **Star this repository if you find it helpful!** ⭐
