# ViKey: Secure Door Access with Passive Light-Based Optical Tags

**ViKey** is a low-cost, passive authentication system for door access control using visible light backscatter. It employs 3D position-dependent optical patterns created by layered transparent tapes and polarizing films. These patterns are invisible to the human eye and captured by a polarized camera, enabling secure, contactless access without biometrics or RF signals.
---

## Repository Structure

| Folder                     | Description                     |
|---------------------------|---------------------------------|
| `assets/`            | inference, register.png |
| `data/`       | accuracy, latency, computing overhead data  |
| `patterns/` | vikey patterns  |
| `scripts/`          | computer-vision workflow    |

---

## Features

- Passive and revocable visible-light authentication
- Infinite key space using layered tape birefringence
- Lightweight computer vision-based pattern recognition
- Resistant to spoofing, cloning, and replay attacks
- Minimal hardware: commercial webcam + polarizer



---

## Tooling

- **Tag Design**: 3D printed base + polarizer + adhesive tape layers
- **Pattern Recognition**: Python (OpenCV, SIFT, CLAHE)
- **UI**: Tkinter for real-time reader interface
- **Camera**: USB webcam with a mounted polarizing film
---

## Getting Started

```bash
git clone https://github.com/JaskiratSudan/ViKey.git
cd scripts
