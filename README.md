# Numbers_Speech

Speech-to-Text for Digits  
Convert spoken digits into text (or numeric) form using machine learning.

---

## Table of Contents

- [About](#about)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Details](#model-details)  
- [Folder Structure](#folder-structure)  
- [Contributing](#contributing)  
- [License](#license)  

---

## About

Numbers_Speech is a Python project that recognizes spoken digits (0-9) from audio and converts them to text (or numeric) output. It can be used in applications that require voice input of digits, e.g. voice-based PIN entry, telephone-style input, or other simple voice interfaces.

---

## Features

- Recognizes individual digit audio samples  
- Uses a trained machine learning / deep learning model (model file included)  
- Simple API / script for inference  
- Light dependencies  

---

## Requirements

- Python 3.x  
- Libraries: (examples; adjust as needed)

  - `numpy`  
  - `scipy` or `librosa` (for audio feature extraction)  
  - `sklearn` or other ML framework used  
  - `soundfile` or `wave` (for audio loading)  

- A trained model (provided as `model.pkl`)  

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/AlbertZaqaryan/Numbers_Speech.git
   cd Numbers_Speech
