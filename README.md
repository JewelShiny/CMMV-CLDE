# CMMV-CLDE

Cross-Modal Multi-View Alignment with Description Enhancement for Street-View Geo-Localization

## Environment Setup

1. Clone this repository and navigate to CMMV-CLDE folder
```bash
git clone https://github.com/2285443514/CMMV-CLDE
cd CMMV-CLDE
```

2. Install Package
```Shell
conda create -n CMMV python=3.10 -y
conda activate CMMV
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Training Data
Data Available: [Baidu Cloud](https://pan.baidu.com/s/1Ldv-KuwoxpTzMCcn8uC4qg?pwd=81pz)


## Training Script

### Stage 1

```Shell
sh run_MVCV.sh
```

### Stage 2

```Shell
sh run_MVCV_text_llm.sh
```

## Acknowledgement
The code base is mainly from the [Sample4Geo](https://github.com/Skyy93/Sample4Geo) project.

