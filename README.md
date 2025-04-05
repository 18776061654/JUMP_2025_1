# Long Jump Motion Analysis System (JumpTest)

A computer vision-based long jump motion analysis system for evaluating and guiding athletes' technical movements.

## Project Overview

This project is a professional long jump motion analysis system that uses computer vision technology to capture and analyze athletes' movements in real-time. The system can automatically identify and evaluate key stages of the long jump process, including takeoff, flight, and landing, and provide detailed scores and feedback.

## Key Features

- 🎯 **Real-time Motion Capture**: Capture athletes' movements in real-time using cameras
- 📊 **Motion Analysis**: Automatically identify and analyze various stages of the long jump
- 📈 **Scoring System**: Score key movements such as takeoff, flight, and landing
- 📝 **Data Recording**: Record data for each training session, supporting multiple test comparisons
- 👥 **Student Management**: Support multi-student management, recording each student's training data
- 📱 **User Interface**: Intuitive graphical interface for easy operation and result viewing

## Technical Features

- Real-time motion capture using OpenCV and MediaPipe
- Computer vision-based motion analysis and evaluation
- Modern graphical interface built with PySide6
- Support for multi-camera synchronized recording
- Automatic video data saving and analysis

## Installation Instructions

### Environment Requirements

- **Operating System**: Windows 10 / Ubuntu 20.04 (dual system testing)
- **Python Version**: Python 3.10.16
- **Hardware Requirements**:
  - Camera: 120° wide-angle camera (1920x1080, 120fps)
  - CPU: Intel® Xeon® Silver 4110 CPU @ 2.10GHz
  - GPU: NVIDIA RTX 3080Ti 12GB
  - Memory: 32GB DDR4
- **Development Environment**: Anaconda 3 + Python 3.10.16 + CUDA 12.4 + cuDNN 8.9 + PyTorch 2.1.1
- **Third-party Libraries**: OpenCV 4.11, NumPy 1.21, Matplotlib 3.5, MMPose 1.3.2, etc.

### Installation Steps

1. Clone the project locally:
```bash
git clone https://github.com/18776061654/JUMP_2025_1.git
cd JumpTest
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the program:
```bash
python main.py
```

## Usage Instructions

1. **Student Management**
   - Import student information
   - Select the student for the current test
   - View historical test records

2. **Testing Process**
   - Select the number of tests (supports 3 tests)
   - Start recording (simultaneously record run-up and long jump movements)
   - The system automatically analyzes movements and generates scores

3. **Result Viewing**
   - View detailed scoring results
   - Compare data from different tests
   - Export test reports

## Project Structure

```
JumpTest/
├── main.py              # Main program entry
├── main_part/           # Main functional modules
├── posture/            # Code related to posture analysis
├── speed/              # Code related to speed analysis
├── student/            # Code related to student management
├── public/             # Public resource files
├── results/            # Test result storage
└── requirements.txt    # Project dependencies
```

## Contribution Guidelines

Feel free to submit Issues and Pull Requests to help improve the project. Before submitting code, please ensure:

1. Code complies with PEP 8 standards
2. Necessary comments are added
3. Relevant documentation is updated



## Contact

For any questions or suggestions, please contact:

- Submit an Issue
- Email: [LuMin510@126.com]

## Acknowledgments

Thanks to all the developers who contributed to this project! 
