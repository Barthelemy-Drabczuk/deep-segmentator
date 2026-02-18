#!/usr/bin/env python3
"""Dataset download instructions and helper (stub)."""
import sys
from pathlib import Path


def print_instructions() -> None:
    print("""
Sulcal Segmentation Dataset Download Instructions
=================================================

1. UKBioBank (40k subjects, 3T Siemens Skyra, ages 40–80)
   - Access: https://www.ukbiobank.ac.uk/researchers (requires application)
   - Data field: 20227 (T1 structural brain MRI)
   - Labels: Morphologist + FreeSurfer pipelines via BrainVISA

2. ABCD (2.5k subjects, multi-vendor, ages 9–11)
   - Access: https://nda.nih.gov/abcd (NDA account required)
   - Download: abcd-dicom2bids tool recommended

3. ABIDE (1.1k subjects, ASD + control, ages 7–64)
   - Access: http://fcon_1000.projects.nitrc.org/indi/abide/ (open access)
   - Preprocessed: http://preprocessed-connectomes-project.org/abide/

4. SENIOR (142 subjects, 3T + 7T)
   - Contact: CEA NeuroSpin data committee
   - Internal access only

After downloading, update configs/datasets/{name}.yaml with correct root_dir paths.
Then run: python scripts/preprocess_all.py
""")


if __name__ == "__main__":
    print_instructions()
